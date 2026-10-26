import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

st.set_page_config(
    page_title="RetailIQ | Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- DESIGN SYSTEM -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#080d18; --panel:#111a2b; --panel2:#151f33; --line:#26334b;
        --muted:#91a0bb; --text:#edf3ff; --cyan:#52d6c7; --blue:#7aa7ff; }
.stApp { background: radial-gradient(ellipse at 10% 0%, #14203b 0%, var(--bg) 42%); color:var(--text); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#0d1525; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:var(--text); }
h1 { font-size:2.25rem !important; font-weight:800 !important; }
h2 { font-size:1.45rem !important; font-weight:750 !important; }
h3 { font-size:1.08rem !important; font-weight:700 !important; }
p, label, .stMarkdown { color:var(--text); }
[data-testid="stMetric"] { background:linear-gradient(145deg,#17243b,#101827); border:1px solid #2b3a56; border-radius:16px; padding:18px 18px 15px; box-shadow:0 8px 24px rgba(0,0,0,.12); }
[data-testid="stMetricLabel"] { color:#a9b8d2 !important; font-size:.83rem !important; }
[data-testid="stMetricValue"] { color:#f5f8ff !important; font-family:'Manrope',sans-serif; font-size:1.7rem !important; font-weight:800; }
[data-testid="stMetricDelta"] { font-size:.78rem !important; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid #354968; background:#16233a; color:#eaf1ff; font-weight:650; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#52d6c7; color:#52d6c7; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#111a2b; border-color:#2b3a56; }
hr { border-color:#26334b; }
.eyebrow { color:#52d6c7; font-size:.75rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.3rem; }
.hero { background:linear-gradient(120deg,rgba(36,68,111,.72),rgba(20,34,55,.78) 58%,rgba(18,65,68,.5)); border:1px solid #2a4561; border-radius:22px; padding:25px 28px; margin:.25rem 0 1.2rem; }
.hero-title { font-family:'Manrope',sans-serif; font-size:1.75rem; font-weight:800; letter-spacing:-.04em; margin:0; color:#f4f7ff; }
.hero-sub { color:#b4c3dd; margin-top:.45rem; font-size:.95rem; }
.section-note { color:#94a4c0; font-size:.84rem; }
.tag { display:inline-block; border:1px solid #2d5360; background:#102c35; color:#71e5d6; border-radius:999px; padding:4px 10px; font-size:.72rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

REQUIRED = [
    "Transaction_ID","Date","Salesman_ID","Salesman_Name","Shop_ID","Shop_Name",
    "City","Product_ID","Product_Name","Category","Quantity","Unit_Price",
    "Discount_Pct","Gross_Sales","Discount_Amount","Revenue","Cost","Profit",
    "Sales_Target","Target_Achievement_Pct"
]
NUMERIC = [
    "Quantity","Unit_Price","Discount_Pct","Gross_Sales","Discount_Amount",
    "Revenue","Cost","Profit","Sales_Target","Target_Achievement_Pct"
]
ALIASES = {c.lower(): c for c in REQUIRED}

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    rename = {c: ALIASES[c.lower()] for c in df.columns if c.lower() in ALIASES}
    return df.rename(columns=rename)

@st.cache_data(show_spinner=False)
def read_csv_cached(path, mtime):
    del mtime
    return pd.read_csv(path)

def load_mysql():
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    database = os.getenv("MYSQL_DATABASE", "retail_sales")
    username = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    table = os.getenv("MYSQL_TABLE", "sales_transactions")
    if not password:
        raise ValueError("MYSQL_PASSWORD is empty. Set it in your local .env file.")
    if not table.replace("_", "").isalnum():
        raise ValueError("MYSQL_TABLE may contain only letters, numbers and underscores.")
    url = URL.create("mysql+pymysql", username=username, password=password,
                     host=host, port=port, database=database)
    engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 8})
    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(f"SELECT * FROM `{table}`"), conn)
    finally:
        engine.dispose()
    return result

def prep(df):
    df = normalize_columns(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Revenue", "Profit"]).copy()
    for c in ["Transaction_ID","Salesman_Name","Shop_Name","City","Category","Product_Name"]:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df

def money(v):
    return f"₹{v:,.0f}"

def pct(v):
    return f"{v:.1f}%"

def chart_style(fig, height=360):
    # Keep titles, legends and plotting area in separate vertical zones.
    # This prevents Plotly titles from colliding with multi-item legends.
    has_title = bool(fig.layout.title and fig.layout.title.text)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial, sans-serif", color="#dce6fb", size=12),
        margin=dict(l=18, r=18, t=(112 if has_title else 56), b=24), height=height,
        title=dict(x=0.02, xanchor="left", y=0.99, yanchor="top",
                   font=dict(size=17, color="#f4f7ff")) if has_title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="left", x=0, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", itemwidth=70),
        hoverlabel=dict(bgcolor="#101a2b", font_color="#f4f7ff"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#26334b", automargin=True)
    fig.update_yaxes(gridcolor="rgba(145,165,205,.13)", zeroline=False, automargin=True)
    return fig

def section(title, note=None):
    st.markdown(f"## {title}")
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)

def kpis(df):
    revenue = df["Revenue"].sum()
    profit = df["Profit"].sum()
    cost = df["Cost"].sum()
    target = df["Sales_Target"].sum()
    gross = df["Gross_Sales"].sum()
    units = df["Quantity"].sum()
    discount = df["Discount_Amount"].sum()
    margin = profit / revenue * 100 if revenue else 0
    achievement = revenue / target * 100 if target else 0
    cols = st.columns(6)
    cols[0].metric("NET REVENUE", money(revenue))
    cols[1].metric("TOTAL PROFIT", money(profit))
    cols[2].metric("PROFIT MARGIN", pct(margin))
    cols[3].metric("UNITS SOLD", f"{units:,.0f}")
    cols[4].metric("TARGET ACHIEVEMENT", pct(achievement))
    cols[5].metric("DISCOUNTS GIVEN", money(discount))
    return {"revenue":revenue,"profit":profit,"cost":cost,"target":target,
            "gross":gross,"units":units,"discount":discount,"margin":margin,
            "achievement":achievement}

def daily_rollup(df):
    # Keep Date as an explicit column to avoid pandas groupby FutureWarning.
    return (df.assign(Date=df["Date"].dt.date)
            .groupby("Date", as_index=False)
            .agg(Revenue=("Revenue","sum"), Profit=("Profit","sum"),
                 Transactions=("Transaction_ID","count"), Units=("Quantity","sum")))

# --------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">RETAIL ANALYTICS SUITE</div>', unsafe_allow_html=True)
    st.markdown("### RetailIQ")
    st.caption("Decision-ready sales intelligence")
    data_source = st.radio("DATA SOURCE", ["Included CSV dataset","MySQL database"], index=0)
    st.divider()

try:
    if data_source == "Included CSV dataset":
        csv_path = Path(__file__).parent / "data" / "retail_sales_transactions.csv"
        if csv_path.exists():
            raw = read_csv_cached(str(csv_path), csv_path.stat().st_mtime)
        else:
            with st.sidebar:
                uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
            if uploaded is None:
                st.error("Dataset not found. Upload a CSV file to continue.")
                st.stop()
            raw = pd.read_csv(uploaded)
    else:
        with st.spinner("Loading data from MySQL..."):
            raw = load_mysql()
    data = prep(raw)
except Exception as exc:
    st.error("Could not load the selected data source.")
    st.exception(exc)
    if data_source == "MySQL database":
        st.info("Check MySQL service status and the connection values in your .env file.")
    st.stop()

if data.empty:
    st.warning("No valid transaction records found.")
    st.stop()

min_date, max_date = data["Date"].min().date(), data["Date"].max().date()
with st.sidebar:
    st.divider()
    st.markdown("### Global filters")
    date_value = st.date_input("Date range", value=(min_date,max_date),
                               min_value=min_date,max_value=max_date)
    cities = sorted(data["City"].unique().tolist())
    categories = sorted(data["Category"].unique().tolist())
    sellers = sorted(data["Salesman_Name"].unique().tolist())
    shops = sorted(data["Shop_Name"].unique().tolist())
    chosen_cities = st.multiselect("Cities", cities, default=cities)
    chosen_categories = st.multiselect("Categories", categories, default=categories)
    chosen_sellers = st.multiselect("Salespeople", sellers, default=sellers)
    st.caption(f"{len(data):,} records loaded")

if isinstance(date_value, (tuple,list)) and len(date_value) == 2:
    start, end = pd.Timestamp(date_value[0]), pd.Timestamp(date_value[1])
else:
    start, end = pd.Timestamp(min_date), pd.Timestamp(max_date)
filtered = data.loc[
    data["Date"].ge(start) & data["Date"].lt(end + pd.Timedelta(days=1))
    & data["City"].isin(chosen_cities)
    & data["Category"].isin(chosen_categories)
    & data["Salesman_Name"].isin(chosen_sellers)
].copy()

if filtered.empty:
    st.warning("No records match these filters. Adjust the global filters in the sidebar.")
    st.stop()

# Page navigation
page_options = [
    "Executive Overview", "Sales Trends", "Product & Category",
    "Cities & Stores", "Sales Team", "Profitability & Discounts",
    "Transaction Explorer", "ML Insights", "Data Quality"
]
with st.sidebar:
    st.divider()
    page = st.radio("NAVIGATION", page_options, index=0)
    st.caption("RetailIQ • Analytics workspace")

# Shared data and header
st.markdown('<div class="eyebrow">RETAIL PERFORMANCE / 2026</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero"><div class="hero-title">{page}</div><div class="hero-sub">A detailed view of your retail performance, powered by {len(filtered):,} filtered transactions.</div></div>', unsafe_allow_html=True)

# ================================ PAGE 1 ====================================
if page == "Executive Overview":
    kpis(filtered)
    left, right = st.columns([1.55,1])
    with left:
        section("Revenue & profit trajectory", "Daily financial performance across the selected period.")
        daily = daily_rollup(filtered)
        fig = px.area(daily, x="Date", y="Revenue", title="Daily revenue", markers=False)
        fig.update_traces(line_color="#52d6c7", fillcolor="rgba(82,214,199,.15)")
        fig.add_scatter(x=daily["Date"], y=daily["Profit"], mode="lines",
                        name="Profit", line=dict(color="#7aa7ff",width=2.5))
        st.plotly_chart(chart_style(fig,390),use_container_width=True)
    with right:
        section("Revenue mix", "Contribution by product category.")
        cat = filtered.groupby("Category",as_index=False)["Revenue"].sum().sort_values("Revenue",ascending=False)
        fig = px.pie(cat,names="Category",values="Revenue",hole=.68,title="Category share")
        fig.update_traces(textposition="inside",textinfo="percent",marker=dict(line=dict(color="#0b1020",width=2)))
        st.plotly_chart(chart_style(fig,390),use_container_width=True)
    section("Business leaders", "Highest revenue cities, products and salespeople in the current selection.")
    a,b,c = st.columns(3)
    with a:
        top = filtered.groupby("City",as_index=False)["Revenue"].sum().nlargest(5,"Revenue").sort_values("Revenue")
        fig=px.bar(top,x="Revenue",y="City",orientation="h",title="Top cities",color="Revenue",color_continuous_scale="Teal")
        st.plotly_chart(chart_style(fig,300),use_container_width=True)
    with b:
        top=filtered.groupby("Product_Name",as_index=False)["Revenue"].sum().nlargest(5,"Revenue").sort_values("Revenue")
        fig=px.bar(top,x="Revenue",y="Product_Name",orientation="h",title="Top products",color="Revenue",color_continuous_scale="Blues")
        st.plotly_chart(chart_style(fig,300),use_container_width=True)
    with c:
        top=filtered.groupby("Salesman_Name",as_index=False)["Revenue"].sum().nlargest(5,"Revenue").sort_values("Revenue")
        fig=px.bar(top,x="Revenue",y="Salesman_Name",orientation="h",title="Top salespeople",color="Revenue",color_continuous_scale="Viridis")
        st.plotly_chart(chart_style(fig,300),use_container_width=True)
    section("Quick readout")
    total_target=filtered["Sales_Target"].sum()
    achieved=filtered["Revenue"].sum()/total_target*100 if total_target else 0
    best_city=filtered.groupby("City")["Revenue"].sum().idxmax()
    best_product=filtered.groupby("Product_Name")["Revenue"].sum().idxmax()
    best_seller=filtered.groupby("Salesman_Name")["Revenue"].sum().idxmax()
    r1,r2,r3=st.columns(3)
    r1.info(f"**Leading city**\n\n{best_city}")
    r2.info(f"**Leading product**\n\n{best_product}")
    r3.info(f"**Leading salesperson**\n\n{best_seller}")
    if achieved < 100:
        st.warning(f"Revenue is at {achieved:.1f}% of the summed sales target for the selected records.")
    else:
        st.success(f"Revenue is at {achieved:.1f}% of the summed sales target for the selected records.")

# ================================ PAGE 2 ====================================
elif page == "Sales Trends":
    kpis(filtered)
    daily=daily_rollup(filtered)
    section("Trend explorer","Inspect revenue, profit, transaction volume and unit volume over time.")
    metric_choice=st.selectbox("Primary trend metric",["Revenue","Profit","Transactions","Units"])
    fig=px.line(daily,x="Date",y=metric_choice,markers=True,title=f"Daily {metric_choice.lower()} trend")
    fig.update_traces(line=dict(color="#52d6c7",width=3),marker=dict(size=5))
    st.plotly_chart(chart_style(fig,420),use_container_width=True)
    left,right=st.columns(2)
    with left:
        monthly=filtered.assign(Month=filtered["Date"].dt.to_period("M").astype(str)).groupby("Month",as_index=False).agg(
            Revenue=("Revenue","sum"),Profit=("Profit","sum"),Transactions=("Transaction_ID","count"))
        fig=px.bar(monthly,x="Month",y=["Revenue","Profit"],barmode="group",title="Monthly revenue vs profit")
        st.plotly_chart(chart_style(fig,350),use_container_width=True)
    with right:
        weekday=filtered.assign(Weekday=filtered["Date"].dt.day_name()).groupby("Weekday",as_index=False).agg(
            Revenue=("Revenue","sum"),Transactions=("Transaction_ID","count"))
        order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekday["Weekday"]=pd.Categorical(weekday["Weekday"],categories=order,ordered=True)
        weekday=weekday.sort_values("Weekday")
        fig=px.bar(weekday,x="Weekday",y="Revenue",title="Revenue by day of week",color="Revenue",color_continuous_scale="Teal")
        st.plotly_chart(chart_style(fig,350),use_container_width=True)
    st.dataframe(monthly,use_container_width=True,hide_index=True)

# ================================ PAGE 3 ====================================
elif page == "Product & Category":
    kpis(filtered)
    section("Category scorecard","Compare category revenue, profitability, volume and discount levels.")
    category=filtered.groupby("Category",as_index=False).agg(
        Revenue=("Revenue","sum"),Profit=("Profit","sum"),Cost=("Cost","sum"),
        Units=("Quantity","sum"),Discount=("Discount_Amount","sum"),Transactions=("Transaction_ID","count"))
    category["Margin %"]=category["Profit"].div(category["Revenue"].replace(0,pd.NA))*100
    category=category.sort_values("Revenue",ascending=False)
    left,right=st.columns(2)
    with left:
        fig=px.bar(category.sort_values("Revenue"),x="Revenue",y="Category",orientation="h",
                   color="Margin %",color_continuous_scale="Tealgrn",title="Revenue and margin by category",
                   hover_data=["Profit","Units","Discount"])
        st.plotly_chart(chart_style(fig,430),use_container_width=True)
    with right:
        fig=px.scatter(category,x="Revenue",y="Profit",size="Units",color="Category",
                       hover_name="Category",title="Category portfolio: revenue vs profit",size_max=48)
        st.plotly_chart(chart_style(fig,430),use_container_width=True)
    section("Product leaderboard","Rank products by revenue, profit or units sold.")
    rank_by=st.selectbox("Rank products by",["Revenue","Profit","Quantity","Discount_Amount"],key="product_rank")
    product=filtered.groupby(["Product_ID","Product_Name","Category"],as_index=False).agg(
        Revenue=("Revenue","sum"),Profit=("Profit","sum"),Quantity=("Quantity","sum"),
        Discount_Amount=("Discount_Amount","sum"),Transactions=("Transaction_ID","count"))
    product=product.nlargest(15,rank_by).sort_values(rank_by,ascending=True)
    fig=px.bar(product,x=rank_by,y="Product_Name",orientation="h",color=rank_by,
               color_continuous_scale="Viridis",title=f"Top 15 products by {rank_by.replace('_',' ').lower()}")
    st.plotly_chart(chart_style(fig,500),use_container_width=True)
    st.dataframe(category,use_container_width=True,hide_index=True)

# ================================ PAGE 4 ====================================
elif page == "Cities & Stores":
    kpis(filtered)
    section("Geographic performance","Compare city and store contribution to revenue and profit.")
    city=filtered.groupby("City",as_index=False).agg(Revenue=("Revenue","sum"),Profit=("Profit","sum"),
        Units=("Quantity","sum"),Transactions=("Transaction_ID","count"),Discount=("Discount_Amount","sum"))
    city["Margin %"]=city["Profit"].div(city["Revenue"].replace(0,pd.NA))*100
    left,right=st.columns(2)
    with left:
        fig=px.bar(city.sort_values("Revenue"),x="Revenue",y="City",orientation="h",
                   color="Profit",color_continuous_scale="Teal",title="Revenue by city",
                   hover_data=["Profit","Units","Transactions"])
        st.plotly_chart(chart_style(fig,440),use_container_width=True)
    with right:
        fig=px.scatter(city,x="Revenue",y="Profit",size="Transactions",color="Margin %",
                       hover_name="City",size_max=40,color_continuous_scale="RdYlGn",
                       title="City portfolio: revenue vs profit")
        st.plotly_chart(chart_style(fig,440),use_container_width=True)
    section("Store leaderboard")
    store=filtered.groupby(["Shop_ID","Shop_Name","City"],as_index=False).agg(
        Revenue=("Revenue","sum"),Profit=("Profit","sum"),Units=("Quantity","sum"),
        Transactions=("Transaction_ID","count"))
    store=store.sort_values("Revenue",ascending=False)
    topn=st.slider("Number of stores to display",5,min(50,max(5,len(store))),15)
    fig=px.bar(store.head(topn).sort_values("Revenue"),x="Revenue",y="Shop_Name",orientation="h",
               color="City",hover_data=["Profit","Units","Transactions"],title=f"Top {topn} stores")
    st.plotly_chart(chart_style(fig,500),use_container_width=True)
    st.dataframe(store,use_container_width=True,hide_index=True)

# ================================ PAGE 5 ====================================
elif page == "Sales Team":
    kpis(filtered)
    section("Salesforce performance","Compare seller results against targets and contribution.")
    seller=filtered.groupby(["Salesman_ID","Salesman_Name"],as_index=False).agg(
        Revenue=("Revenue","sum"),Profit=("Profit","sum"),Target=("Sales_Target","sum"),
        Transactions=("Transaction_ID","count"),Units=("Quantity","sum"))
    seller["Target Achievement %"]=seller["Revenue"].div(seller["Target"].replace(0,pd.NA))*100
    seller["Profit Margin %"]=seller["Profit"].div(seller["Revenue"].replace(0,pd.NA))*100
    left,right=st.columns(2)
    with left:
        fig=px.bar(seller.nlargest(15,"Revenue").sort_values("Revenue"),x="Revenue",y="Salesman_Name",
                   orientation="h",color="Target Achievement %",color_continuous_scale="RdYlGn",
                   title="Top salespeople by revenue",hover_data=["Profit","Target Achievement %","Transactions"])
        st.plotly_chart(chart_style(fig,500),use_container_width=True)
    with right:
        fig=px.scatter(seller,x="Target Achievement %",y="Profit",size="Revenue",color="Salesman_Name",
                       hover_name="Salesman_Name",size_max=38,title="Target achievement vs profit")
        fig.add_vline(x=100,line_dash="dash",line_color="#52d6c7",annotation_text="100% target")
        st.plotly_chart(chart_style(fig,500),use_container_width=True)
    st.dataframe(seller.sort_values("Revenue",ascending=False),use_container_width=True,hide_index=True)

# ================================ PAGE 6 ====================================
elif page == "Profitability & Discounts":
    kpis(filtered)
    section("Profitability diagnostics","Explore gross-to-net movement, costs, margin and discount intensity.")
    rev=filtered["Revenue"].sum()
    gross=filtered["Gross_Sales"].sum()
    discount=filtered["Discount_Amount"].sum()
    cost=filtered["Cost"].sum()
    profit=filtered["Profit"].sum()
    a,b,c,d=st.columns(4)
    a.metric("Gross sales",money(gross))
    b.metric("Discount amount",money(discount))
    c.metric("Cost base",money(cost))
    d.metric("Net profit",money(profit))
    left,right=st.columns(2)
    with left:
        bridge=pd.DataFrame({"Stage":["Gross sales","Discounts","Cost","Profit"],
                             "Amount":[gross,-discount,-cost,profit]})
        fig=go.Figure(go.Waterfall(
            x=bridge["Stage"],y=bridge["Amount"],measure=["absolute","relative","relative","total"],
            text=[money(v) for v in bridge["Amount"]],textposition="outside",
            increasing={"marker":{"color":"#52d6c7"}},decreasing={"marker":{"color":"#ff7f96"}},
            totals={"marker":{"color":"#7aa7ff"}}))
        fig.update_layout(title="Revenue-to-profit bridge",showlegend=False)
        st.plotly_chart(chart_style(fig,400),use_container_width=True)
    with right:
        disc=filtered.groupby("Category",as_index=False).agg(
            Revenue=("Revenue","sum"),Discount=("Discount_Amount","sum"),Profit=("Profit","sum"))
        disc["Discount % of gross"]=disc["Discount"].div(
            (disc["Revenue"]+disc["Discount"]).replace(0,pd.NA))*100
        fig=px.scatter(disc,x="Discount % of gross",y="Profit",size="Revenue",color="Category",
                       hover_name="Category",size_max=42,title="Discount intensity vs profit")
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.20,
                                      xanchor="left", x=0, font=dict(size=10),
                                      bgcolor="rgba(0,0,0,0)"), margin=dict(l=18,r=18,t=90,b=75))
        st.plotly_chart(chart_style(fig,430),use_container_width=True)
    section("Margin by product")
    prod=filtered.groupby(["Product_Name","Category"],as_index=False).agg(
        Revenue=("Revenue","sum"),Profit=("Profit","sum"),Discount=("Discount_Amount","sum"))
    prod["Margin %"]=prod["Profit"].div(prod["Revenue"].replace(0,pd.NA))*100
    prod=prod.sort_values("Revenue",ascending=False)
    fig=px.bar(prod.head(20).sort_values("Margin %"),x="Margin %",y="Product_Name",orientation="h",
               color="Margin %",color_continuous_scale="RdYlGn",title="Profit margin for top 20 revenue products")
    st.plotly_chart(chart_style(fig,520),use_container_width=True)
    st.dataframe(disc.sort_values("Profit",ascending=False),use_container_width=True,hide_index=True)

# ================================ PAGE 7 ====================================
elif page == "Transaction Explorer":
    kpis(filtered)
    section("Transaction-level drill-down","Search, sort, inspect and export the filtered transaction records.")
    f1,f2,f3=st.columns([1.2,1,1])
    with f1:
        search=st.text_input("Search all fields",placeholder="Transaction ID, shop, product...")
    with f2:
        min_revenue,max_revenue=float(filtered["Revenue"].min()),float(filtered["Revenue"].max())
        revenue_range=st.slider("Revenue per transaction",min_revenue,max_revenue,
                                (min_revenue,max_revenue))
    with f3:
        sort_column=st.selectbox("Sort table by",["Date","Revenue","Profit","Quantity","Discount_Amount"])
    table=filtered.loc[filtered["Revenue"].between(revenue_range[0],revenue_range[1])].copy()
    if search.strip():
        q=search.strip().lower()
        mask=table.astype(str).apply(lambda col:col.str.lower().str.contains(q,regex=False,na=False)).any(axis=1)
        table=table.loc[mask]
    table=table.sort_values(sort_column,ascending=False)
    st.caption(f"{len(table):,} records match the current explorer controls.")
    st.dataframe(table,use_container_width=True,hide_index=True,height=520)
    st.download_button("⬇️ Export filtered records (CSV)",table.to_csv(index=False).encode("utf-8"),
                       "retail_transactions_export.csv","text/csv")
    st.download_button("⬇️ Export summary (CSV)",
                       filtered.groupby(["Date","City","Category"],as_index=False).agg(
                           Revenue=("Revenue","sum"),Profit=("Profit","sum"),
                           Units=("Quantity","sum")).to_csv(index=False).encode("utf-8"),
                       "retail_summary_export.csv","text/csv")

# ================================ ML INSIGHTS ===============================
elif page == "ML Insights":
    st.markdown("### Machine learning workspace")
    st.caption("Models run locally on the selected data. Forecasts are estimates, not guarantees.")
    tab_forecast, tab_segments = st.tabs(["📈 Sales Forecasting", "🧩 Salesperson Segmentation"])

    with tab_forecast:
        section("Daily revenue forecast", "A transparent Linear Regression baseline with calendar features and a chronological holdout test.")
        horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=30, step=1)
        daily = filtered.assign(Date=filtered["Date"].dt.floor("D")).groupby("Date", as_index=False).agg(Revenue=("Revenue", "sum"))
        daily = daily.sort_values("Date").dropna(subset=["Revenue"])
        if len(daily) < 14:
            st.warning("At least 14 distinct days are needed to train and evaluate the forecast. Expand the date filter.")
        else:
            def forecast_features(dates, origin):
                dates = pd.to_datetime(pd.Series(dates))
                return pd.DataFrame({
                    "day_index": (dates - origin).dt.days.astype(float),
                    "weekday": dates.dt.dayofweek.astype(int),
                    "month": dates.dt.month.astype(int),
                    "day_of_month": dates.dt.day.astype(int),
                })
            origin = daily["Date"].min()
            X = forecast_features(daily["Date"], origin)
            y = daily["Revenue"].astype(float).to_numpy()
            split = max(1, int(len(daily) * 0.8))
            if split >= len(daily):
                split = len(daily) - 1
            model = LinearRegression()
            model.fit(X.iloc[:split], y[:split])
            test_pred = np.maximum(0, model.predict(X.iloc[split:]))
            mae = float(mean_absolute_error(y[split:], test_pred))
            rmse = float(np.sqrt(mean_squared_error(y[split:], test_pred)))
            m1, m2, m3 = st.columns(3)
            m1.metric("Holdout MAE", f"₹{mae:,.0f}")
            m2.metric("Holdout RMSE", f"₹{rmse:,.0f}")
            m3.metric("Training days", f"{split:,}")
            st.caption("MAE/RMSE are calculated on the final 20% of dates, keeping the test period later than the training period.")

            full_model = LinearRegression().fit(X, y)
            last_date = daily["Date"].max()
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
            future_X = forecast_features(future_dates, origin)
            future_pred = np.maximum(0, full_model.predict(future_X))
            history = daily.rename(columns={"Revenue": "Actual revenue"}).copy()
            history["Forecast revenue"] = np.nan
            future = pd.DataFrame({"Date": future_dates, "Actual revenue": np.nan, "Forecast revenue": future_pred})
            plot_df = pd.concat([history, future], ignore_index=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history["Date"], y=history["Actual revenue"], name="Actual revenue", mode="lines", line=dict(color="#52d6c7", width=2.5)))
            fig.add_trace(go.Scatter(x=future["Date"], y=future["Forecast revenue"], name="Forecast", mode="lines+markers", line=dict(color="#7aa7ff", width=2.5, dash="dash")))
            fig.update_layout(title="Historical revenue and forward forecast", xaxis_title="Date", yaxis_title="Revenue (₹)")
            st.plotly_chart(chart_style(fig, 430), use_container_width=True)
            forecast_table = future.copy()
            forecast_table["Forecast revenue"] = forecast_table["Forecast revenue"].round(2)
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download forecast CSV", forecast_table.to_csv(index=False).encode("utf-8"), "retail_revenue_forecast.csv", "text/csv")
            st.info("Model limitation: this is a baseline linear model with weekday/month calendar features. It does not know about promotions, holidays, stockouts, or future business changes.")

    with tab_segments:
        section("Salesperson performance segments", "K-Means groups salespeople using revenue, profit, units sold, and target achievement.")
        seller = filtered.groupby(["Salesman_ID", "Salesman_Name"], as_index=False).agg(
            Revenue=("Revenue", "sum"), Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"), Target_Achievement=("Target_Achievement_Pct", "mean"),
            Transactions=("Transaction_ID", "nunique"), Shops=("Shop_ID", "nunique"))
        features = ["Revenue", "Profit", "Quantity", "Target_Achievement"]
        seller_model_data = seller.dropna(subset=features).copy()
        if len(seller_model_data) < 3:
            st.warning("At least three salespeople with complete performance data are needed for segmentation. Expand your filters.")
        else:
            k = st.slider("Number of segments", min_value=2, max_value=min(5, len(seller_model_data)), value=min(3, len(seller_model_data)), key="ml_cluster_count")
            scaler = StandardScaler()
            scaled = scaler.fit_transform(seller_model_data[features])
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            seller_model_data["Segment_ID"] = km.fit_predict(scaled)
            segment_order = seller_model_data.groupby("Segment_ID")["Revenue"].mean().sort_values().index.tolist()
            segment_names = {cluster: ("Developing" if i == 0 else "Established" if i == len(segment_order)-1 else f"Segment {i+1}") for i, cluster in enumerate(segment_order)}
            seller_model_data["Segment"] = seller_model_data["Segment_ID"].map(segment_names)
            a, b = st.columns([1.2, 1])
            with a:
                fig = px.scatter(seller_model_data, x="Revenue", y="Profit", size="Quantity", color="Segment", hover_name="Salesman_Name", hover_data=["Target_Achievement", "Transactions", "Shops"], title="Salespeople grouped by revenue and profit")
                st.plotly_chart(chart_style(fig, 420), use_container_width=True)
            with b:
                segment_summary = seller_model_data.groupby("Segment", as_index=False).agg(
                    Salespeople=("Salesman_ID", "nunique"), Revenue=("Revenue", "sum"), Profit=("Profit", "sum"),
                    Avg_Target_Achievement=("Target_Achievement", "mean"))
                st.dataframe(segment_summary.round(2), use_container_width=True, hide_index=True)
                st.caption("Segment names are relative to the selected data. 'Developing' means the lowest average revenue cluster, not a permanent employee rating.")
            st.dataframe(seller_model_data.sort_values(["Segment", "Revenue"], ascending=[True, False]).drop(columns=["Segment_ID"]), use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download salesperson segments", seller_model_data.to_csv(index=False).encode("utf-8"), "retail_salesperson_segments.csv", "text/csv")

# ================================ PAGE 9 ====================================
elif page == "Data Quality":
    kpis(filtered)
    section("Data quality monitor","Quick checks on missing values, duplicate transactions and numeric validity.")
    total_rows=len(filtered)
    missing=filtered[REQUIRED].isna().sum().rename("Missing values").to_frame()
    duplicate_count=int(filtered["Transaction_ID"].duplicated().sum())
    invalid_qty=int((filtered["Quantity"]<=0).sum())
    invalid_financial=int(((filtered["Revenue"]<0)|(filtered["Cost"]<0)).sum())
    a,b,c,d=st.columns(4)
    a.metric("Rows in selection",f"{total_rows:,}")
    b.metric("Duplicate transaction IDs",f"{duplicate_count:,}")
    c.metric("Non-positive quantities",f"{invalid_qty:,}")
    d.metric("Negative revenue/cost rows",f"{invalid_financial:,}")
    left,right=st.columns(2)
    with left:
        section("Missing values by column")
        missing=missing.sort_values("Missing values",ascending=False).reset_index(names="Column")
        fig=px.bar(missing.head(20).sort_values("Missing values"),x="Missing values",y="Column",
                   orientation="h",title="Top 20 columns by missing values",color="Missing values",
                   color_continuous_scale="Teal")
        st.plotly_chart(chart_style(fig,470),use_container_width=True)
    with right:
        section("Data preview")
        st.dataframe(filtered.head(100),use_container_width=True,hide_index=True,height=470)
    st.download_button("⬇️ Export data quality report (CSV)",
                       missing.to_csv(index=False).encode("utf-8"),
                       "retail_data_quality_report.csv","text/csv")

st.markdown("---")
st.markdown(
    '<div class="section-note">RetailIQ • Streamlit + Plotly + Pandas • '
    'All metrics respond to the global sidebar filters.</div>',
    unsafe_allow_html=True,
)
