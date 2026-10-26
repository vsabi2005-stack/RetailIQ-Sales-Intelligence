# Retail Sales Intelligence Dashboard

A Streamlit + Plotly dashboard built for the supplied retail transactions dataset.

## Features
- KPI cards: revenue, profit, margin, units sold, and revenue versus target
- Date, city, category, and salesperson filters
- Revenue/profit trend
- Category, city, product, and salesperson charts
- Revenue versus profit scatter plot
- Searchable transaction table and CSV download
- Included CSV mode and optional live MySQL mode

## 1. Open the project in VS Code

Extract the ZIP, then open the `retail_streamlit_dashboard` folder in VS Code.

## 2. Create a virtual environment (Windows CMD)

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py` is unavailable, try `python` instead.

## 3. Run with the included CSV first

```cmd
streamlit run app.py
```

The browser should open the dashboard. No MySQL password is needed for CSV mode.

## 4. Connect to MySQL (optional)

First confirm your database and table names. The default settings are:
- Database: `retail_sales`
- Table: `sales_transactions`

Copy `.env.example` to `.env` and set the correct values:

```cmd
copy .env.example .env
notepad .env
```

Update `MYSQL_PASSWORD` locally. Do not share the `.env` file or commit it to GitHub.

In the app sidebar, select **MySQL database**. If your table has a different name, update `MYSQL_TABLE`.
The MySQL table must contain the same columns as the included CSV, such as `Transaction_ID`, `Date`,
`Revenue`, `Profit`, `Cost`, `Category`, `City`, `Salesman_Name`, and `Target_Achievement_Pct`.

## Troubleshooting
- `Access denied`: check your MySQL username/password in `.env`.
- `Unknown database`: update `MYSQL_DATABASE` to the actual database name.
- `Table doesn't exist`: check `MYSQL_TABLE`.
- Missing columns: inspect the table with `DESCRIBE your_table_name;` in a MySQL SQL prompt.
- Port already in use: close the previous Streamlit process or use `streamlit run app.py --server.port 8502`.

## Dataset
The `data/retail_sales_transactions.csv` file is copied from the dataset supplied in the conversation.
