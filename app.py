from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key-in-production'

# ---------------------------------------------------------------------------
# DATABASE CONFIG
# Default: SQLite (zero setup, great for running/demoing the project).
# To use MySQL instead (as planned in the project report), install
# 'pymysql' and 'cryptography', then uncomment the MySQL line below and
# comment out the SQLite line. Example:
#   mysql+pymysql://root:yourpassword@localhost/hostel_mess_db
# ---------------------------------------------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'hostel_mess.db')
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:password@localhost/hostel_mess_db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB upload limit

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')  # student / admin / warden
    hostel_room = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    complaints = db.relationship('Complaint', backref='student', lazy=True)
    feedbacks = db.relationship('MenuFeedback', backref='student', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # Food Quality, Quantity, Hygiene, Timing, Other
    description = db.Column(db.Text, nullable=False)
    photo_filename = db.Column(db.String(200))
    status = db.Column(db.String(20), default='Pending')  # Pending / In Progress / Resolved
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)


class InventoryItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    unit = db.Column(db.String(20), nullable=False)  # kg, litre, packet, etc.
    current_qty = db.Column(db.Float, nullable=False, default=0)
    reorder_threshold = db.Column(db.Float, nullable=False, default=10)
    lead_time_days = db.Column(db.Integer, nullable=False, default=2)  # days needed to restock

    usage_logs = db.relationship('StockUsageLog', backref='item', lazy=True)


class StockUsageLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('inventory_item.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    qty_used = db.Column(db.Float, nullable=False)


class MenuItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    day_of_week = db.Column(db.String(20), nullable=False)  # Monday .. Sunday
    meal_type = db.Column(db.String(20), nullable=False)  # Breakfast/Lunch/Snacks/Dinner
    dish_name = db.Column(db.String(200), nullable=False)


class MenuFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    meal_date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    meal_type = db.Column(db.String(20), nullable=False)
    dish_name = db.Column(db.String(200))
    rating = db.Column(db.Integer, nullable=False)  # 1-5
    comment = db.Column(db.String(300))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------------------------------------------------------------------
# PREDICTIVE RESTOCKING ALGORITHM
# ---------------------------------------------------------------------------
def moving_average_forecast(usages, window=7):
    """Simple moving average of the most recent `window` usage values."""
    if not usages:
        return 0
    recent = usages[-window:]
    return sum(recent) / len(recent)


def linear_regression_forecast(usages):
    """
    Least-squares linear regression on usage-over-time to catch trends
    (e.g. rising consumption due to more students / seasonal change),
    rather than just flat-averaging like the moving average does.
    Returns predicted next-day usage.
    """
    n = len(usages)
    if n == 0:
        return 0
    if n == 1:
        return usages[0]

    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(usages) / n

    numerator = sum((xs[i] - x_mean) * (usages[i] - y_mean) for i in range(n))
    denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return y_mean

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    predicted = slope * n + intercept  # predict for the next time-step
    return max(predicted, 0)  # usage can't be negative


def get_item_forecast(item, days_history=14):
    cutoff = datetime.utcnow().date() - timedelta(days=days_history)
    logs = (StockUsageLog.query
            .filter(StockUsageLog.item_id == item.id, StockUsageLog.date >= cutoff)
            .order_by(StockUsageLog.date.asc())
            .all())
    usages = [log.qty_used for log in logs]

    ma_forecast = moving_average_forecast(usages)
    lr_forecast = linear_regression_forecast(usages)

    # Blend both models for a slightly more robust estimate
    predicted_daily_usage = (ma_forecast + lr_forecast) / 2 if usages else 0

    if predicted_daily_usage > 0:
        days_until_empty = item.current_qty / predicted_daily_usage
    else:
        days_until_empty = float('inf')

    needs_restock = days_until_empty <= item.lead_time_days

    return {
        'item': item,
        'moving_average': round(ma_forecast, 2),
        'regression_forecast': round(lr_forecast, 2),
        'predicted_daily_usage': round(predicted_daily_usage, 2),
        'days_until_empty': round(days_until_empty, 1) if days_until_empty != float('inf') else None,
        'needs_restock': needs_restock,
        'history_points': len(usages),
    }


# ---------------------------------------------------------------------------
# AUTH ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard' if current_user.role in ('admin', 'warden') else 'student_dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form.get('role', 'student')
        hostel_room = request.form.get('hostel_room')

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))

        user = User(name=name, email=email, role=role, hostel_room=hostel_room)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ---------------------------------------------------------------------------
# STUDENT ROUTES
# ---------------------------------------------------------------------------
@app.route('/student/dashboard')
@login_required
def student_dashboard():
    my_complaints = Complaint.query.filter_by(student_id=current_user.id).order_by(Complaint.created_at.desc()).all()
    menu = MenuItem.query.all()
    return render_template('student_dashboard.html', complaints=my_complaints, menu=menu)


@app.route('/student/complaint/new', methods=['GET', 'POST'])
@login_required
def new_complaint():
    if request.method == 'POST':
        category = request.form['category']
        description = request.form['description']
        photo = request.files.get('photo')
        filename = None

        if photo and photo.filename and allowed_file(photo.filename):
            filename = secure_filename(f"{current_user.id}_{datetime.utcnow().timestamp()}_{photo.filename}")
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        complaint = Complaint(student_id=current_user.id, category=category,
                               description=description, photo_filename=filename)
        db.session.add(complaint)
        db.session.commit()
        flash('Complaint submitted.', 'success')
        return redirect(url_for('student_dashboard'))

    return render_template('new_complaint.html')


@app.route('/student/feedback', methods=['GET', 'POST'])
@login_required
def new_feedback():
    if request.method == 'POST':
        feedback = MenuFeedback(
            student_id=current_user.id,
            meal_type=request.form['meal_type'],
            dish_name=request.form.get('dish_name'),
            rating=int(request.form['rating']),
            comment=request.form.get('comment')
        )
        db.session.add(feedback)
        db.session.commit()
        flash('Thanks for your feedback!', 'success')
        return redirect(url_for('student_dashboard'))

    return render_template('new_feedback.html')


# ---------------------------------------------------------------------------
# ADMIN ROUTES
# ---------------------------------------------------------------------------
def admin_required():
    return current_user.role in ('admin', 'warden')


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not admin_required():
        flash('Access denied.', 'danger')
        return redirect(url_for('student_dashboard'))

    total_complaints = Complaint.query.count()
    pending_complaints = Complaint.query.filter_by(status='Pending').count()
    items = InventoryItem.query.all()
    forecasts = [get_item_forecast(item) for item in items]
    alerts = [f for f in forecasts if f['needs_restock']]

    return render_template('admin_dashboard.html',
                            total_complaints=total_complaints,
                            pending_complaints=pending_complaints,
                            forecasts=forecasts,
                            alerts=alerts)


@app.route('/admin/complaints')
@login_required
def view_complaints():
    if not admin_required():
        return redirect(url_for('student_dashboard'))
    status_filter = request.args.get('status')
    query = Complaint.query
    if status_filter:
        query = query.filter_by(status=status_filter)
    complaints = query.order_by(Complaint.created_at.desc()).all()
    return render_template('view_complaints.html', complaints=complaints, status_filter=status_filter)


@app.route('/admin/complaints/<int:complaint_id>/status', methods=['POST'])
@login_required
def update_complaint_status(complaint_id):
    if not admin_required():
        return redirect(url_for('student_dashboard'))
    complaint = Complaint.query.get_or_404(complaint_id)
    complaint.status = request.form['status']
    if complaint.status == 'Resolved':
        complaint.resolved_at = datetime.utcnow()
    db.session.commit()
    flash('Complaint status updated.', 'success')
    return redirect(url_for('view_complaints'))


@app.route('/admin/inventory', methods=['GET', 'POST'])
@login_required
def inventory():
    if not admin_required():
        return redirect(url_for('student_dashboard'))

    if request.method == 'POST':
        item = InventoryItem(
            name=request.form['name'],
            unit=request.form['unit'],
            current_qty=float(request.form['current_qty']),
            reorder_threshold=float(request.form['reorder_threshold']),
            lead_time_days=int(request.form['lead_time_days'])
        )
        db.session.add(item)
        db.session.commit()
        flash('Item added to inventory.', 'success')
        return redirect(url_for('inventory'))

    items = InventoryItem.query.all()
    return render_template('inventory.html', items=items)


@app.route('/admin/inventory/<int:item_id>/log_usage', methods=['POST'])
@login_required
def log_usage(item_id):
    if not admin_required():
        return redirect(url_for('student_dashboard'))
    item = InventoryItem.query.get_or_404(item_id)
    qty = float(request.form['qty_used'])
    log_date = request.form.get('date')
    log_date = datetime.strptime(log_date, '%Y-%m-%d').date() if log_date else datetime.utcnow().date()

    log = StockUsageLog(item_id=item.id, date=log_date, qty_used=qty)
    item.current_qty = max(item.current_qty - qty, 0)
    db.session.add(log)
    db.session.commit()
    flash(f'Logged usage for {item.name}.', 'success')
    return redirect(url_for('inventory'))


@app.route('/admin/inventory/<int:item_id>/restock', methods=['POST'])
@login_required
def restock_item(item_id):
    if not admin_required():
        return redirect(url_for('student_dashboard'))
    item = InventoryItem.query.get_or_404(item_id)
    qty = float(request.form['qty_added'])
    item.current_qty += qty
    db.session.commit()
    flash(f'Restocked {item.name} by {qty} {item.unit}.', 'success')
    return redirect(url_for('inventory'))


@app.route('/admin/menu', methods=['GET', 'POST'])
@login_required
def manage_menu():
    if not admin_required():
        return redirect(url_for('student_dashboard'))

    if request.method == 'POST':
        menu_item = MenuItem(
            day_of_week=request.form['day_of_week'],
            meal_type=request.form['meal_type'],
            dish_name=request.form['dish_name']
        )
        db.session.add(menu_item)
        db.session.commit()
        flash('Menu item added.', 'success')
        return redirect(url_for('manage_menu'))

    menu = MenuItem.query.all()
    return render_template('manage_menu.html', menu=menu)


@app.route('/admin/analytics')
@login_required
def analytics():
    if not admin_required():
        return redirect(url_for('student_dashboard'))

    # Complaint counts by category
    categories = ['Food Quality', 'Quantity', 'Hygiene', 'Timing', 'Other']
    complaint_counts = [Complaint.query.filter_by(category=c).count() for c in categories]

    # Average rating per dish (top 10 by feedback count)
    feedback_data = {}
    for fb in MenuFeedback.query.all():
        key = fb.dish_name or 'Unknown'
        feedback_data.setdefault(key, []).append(fb.rating)
    dish_names = list(feedback_data.keys())
    dish_avg_ratings = [round(sum(v) / len(v), 2) for v in feedback_data.values()]

    # Inventory forecast summary
    items = InventoryItem.query.all()
    forecasts = [get_item_forecast(item) for item in items]

    return render_template('analytics.html',
                            categories=categories,
                            complaint_counts=complaint_counts,
                            dish_names=dish_names,
                            dish_avg_ratings=dish_avg_ratings,
                            forecasts=forecasts)


@app.route('/api/forecast/<int:item_id>')
@login_required
def api_forecast(item_id):
    item = InventoryItem.query.get_or_404(item_id)
    data = get_item_forecast(item)
    data['item'] = {'id': item.id, 'name': item.name, 'unit': item.unit, 'current_qty': item.current_qty}
    return jsonify(data)


# ---------------------------------------------------------------------------
# CLI helper: create tables + seed an admin account
# ---------------------------------------------------------------------------
@app.cli.command('init-db')
def init_db():
    """Run with: flask --app app init-db"""
    db.create_all()
    if not User.query.filter_by(email='admin@hostel.com').first():
        admin = User(name='Mess Admin', email='admin@hostel.com', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print('Created default admin -> email: admin@hostel.com | password: admin123')
    print('Database initialized.')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='admin@hostel.com').first():
            admin = User(name='Mess Admin', email='admin@hostel.com', role='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
    app.run(debug=True)
