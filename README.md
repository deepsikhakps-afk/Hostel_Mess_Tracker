# Smart Hostel Mess Management System with Predictive Restocking

An MCA-level full-stack project (Flask + SQLAlchemy + Chart.js) that combines:
- **Complaint management** for hostel mess issues (food quality, hygiene, timing, etc.)
- **Inventory tracking** with daily usage logs
- **Predictive restocking** using a moving-average + linear-regression forecasting
  algorithm — the "unique" DAA-relevant core of the project
- **Menu & feedback system** with dish ratings
- **Analytics dashboard** with Chart.js visualizations

## Quick Start

```bash
cd hostel-mess-tracker
pip install -r requirements.txt

# Option A: just run it (auto-creates DB + admin account)
python app.py

# Option B: use Flask CLI
flask --app app init-db
flask --app app run
```

Then open **http://127.0.0.1:5000**

**Default admin login:** `admin@hostel.com` / `admin123`

## Switching to MySQL (optional)

By default the app uses SQLite (zero setup — good for demos/viva). To use MySQL
as originally planned in the project report:

1. `pip install pymysql cryptography`
2. In `app.py`, comment out the SQLite line and uncomment the MySQL line:
   ```python
   app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:yourpassword@localhost/hostel_mess_db'
   ```
3. Create the database first: `CREATE DATABASE hostel_mess_db;`
4. Run `flask --app app init-db`

## Project Structure

```
hostel-mess-tracker/
├── app.py                  # All routes, models, and the prediction algorithm
├── requirements.txt
├── templates/               # Jinja2 HTML templates
│   ├── base.html
│   ├── login.html / register.html
│   ├── student_dashboard.html
│   ├── new_complaint.html / new_feedback.html
│   ├── admin_dashboard.html
│   ├── view_complaints.html
│   ├── inventory.html
│   ├── manage_menu.html
│   └── analytics.html
└── static/
    ├── css/style.css
    └── uploads/              # complaint photos land here
```

## How the Predictive Restocking Algorithm Works

For each inventory item, the system looks at the last 14 days of logged usage
and computes two forecasts:

1. **Moving Average** — average of the last 7 recorded usage values.
   Simple, stable, good baseline.
2. **Linear Regression (least squares)** — fits a trend line through the
   usage history and projects the next day's usage. This catches *rising or
   falling* consumption trends (e.g. more students moving in, seasonal
   variation) that a flat average would miss.

The two are blended (averaged) into a `predicted_daily_usage`. From there:

```
days_until_empty = current_qty / predicted_daily_usage
if days_until_empty <= lead_time_days:
    trigger restock alert
```

This logic lives in `get_item_forecast()` in `app.py` — point to that function
in your project report/viva as the algorithmic core (ties directly into your
DAA coursework: it's a real application of regression-based forecasting, not
just a CRUD form).

## User Roles

| Role    | Capabilities |
|---------|-------------|
| Student | Raise complaints, view menu, rate meals |
| Admin/Warden | Manage complaints, inventory, menu, view analytics & restock alerts |

## Suggested Report Sections (for your synopsis/viva)

1. Introduction & Problem Statement (manual mess tracking is inefficient, stockouts are common)
2. Literature Survey (existing complaint-box systems lack predictive capability)
3. System Design (ER diagram from the models below, DFDs for complaint & restock flow)
4. Algorithm (moving average + linear regression — explain the math, show a sample table)
5. Implementation (Flask/SQLAlchemy backend, Jinja2 + Chart.js frontend)
6. Testing (unit test the forecast function with sample usage data — see below)
7. Results & Screenshot
8. Future Scope (SMS/email alerts, occupancy-adjusted forecasting, mobile app)

# Screenshots
![alt text](image.png)

## Extending It Further

- Add occupancy count as a variable so predicted usage scales with hostel headcount
- Email/SMS alerts when restock is triggered (use Flask-Mail or Twilio)
- Export analytics as PDF report
- Add a "most disliked dish" auto-flag if avg rating < 2.5
