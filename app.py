import os
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory
import requests_cache

# --- Flask App Initialization & Caching Setup ---
app = Flask(__name__)
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
session = requests_cache.CachedSession('yfinance.cache', expire_after=timedelta(hours=1), backend='sqlite')
session.headers['User-Agent'] = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'

# --- Data for Front-End & Peer Analysis ---
SECTOR_STOCK_MAPPING = {
    "Banking & Financials": [{"name": "HDFC Bank", "ticker": "HDFCBANK.NS"}, {"name": "ICICI Bank", "ticker": "ICICIBANK.NS"}, {"name": "State Bank of India", "ticker": "SBIN.NS"}, {"name": "Kotak Mahindra Bank", "ticker": "KOTAKBANK.NS"}, {"name": "Axis Bank", "ticker": "AXISBANK.NS"}, {"name": "Bajaj Finance", "ticker": "BAJFINANCE.NS"}],
    "IT": [{"name": "Infosys", "ticker": "INFY.NS"}, {"name": "TCS", "ticker": "TCS.NS"}, {"name": "Wipro", "ticker": "WIPRO.NS"}, {"name": "Tech Mahindra", "ticker": "TECHM.NS"}, {"name": "HCL Technologies", "ticker": "HCLTECH.NS"}],
    "Energy": [{"name": "Reliance Industries", "ticker": "RELIANCE.NS"}, {"name": "ONGC", "ticker": "ONGC.NS"}, {"name": "NTPC", "ticker": "NTPC.NS"}, {"name": "Power Grid", "ticker": "POWERGRID.NS"}, {"name": "Adani Power", "ticker": "ADANIPOWER.NS"}, {"name": "Tata Power", "ticker": "TATAPOWER.NS"}],
    "Automobiles": [{"name": "Tata Motors", "ticker": "TATAMOTORS.NS"}, {"name": "Mahindra and Mahindra", "ticker": "M&M.NS"}, {"name": "Maruti Suzuki", "ticker": "MARUTI.NS"}, {"name": "Eicher Motors", "ticker": "EICHERMOT.NS"}, {"name": "Bajaj Auto", "ticker": "BAJAJ-AUTO.NS"}]
}
PEER_MAPPING = {
    "HDFCBANK.NS": ["ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"], "ICICIBANK.NS": ["HDFCBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS"], "SBIN.NS": ["HDFCBANK.NS", "ICICIBANK.NS", "PNB.NS", "BANKBARODA.NS"],
    "INFY.NS": ["TCS.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS"], "TCS.NS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "RELIANCE.NS": ["ONGC.NS", "TATAPOWER.NS", "ADANIPOWER.NS"], "TATAMOTORS.NS": ["MARUTI.NS", "M&M.NS", "EICHERMOT.NS"]
}

# ======================================================================
# DATA FETCHING AND ANALYSIS FUNCTIONS (DEFINITIVE VERSION)
# ======================================================================

def fetch_company_info(symbol):
    try:
        ticker = yf.Ticker(symbol, session=session)
        info = ticker.info
        if not info or info.get('marketCap') is None: raise ValueError("Incomplete data from source")
        
        data = {
            'info': info, 'logo_image': fetch_logo(info),
            'financials': ticker.financials, 'balance_sheet': ticker.balance_sheet, 'quarterly_financials': ticker.quarterly_financials,
            'Symbol': symbol, 'Company Name': info.get('longName', 'N/A'), 'Description': info.get('longBusinessSummary', 'N/A'),
            'Market Cap': f"₹{info.get('marketCap', 0) / 1e7:,.2f} Cr", 'Current Price': f"₹{info.get('currentPrice', 'N/A')}",
            'Trailing PE': info.get('trailingPE'), 'ROE': info.get('returnOnEquity'), 'Debt to Equity': info.get('debtToEquity'),
            'Dividend Yield': info.get('dividendYield'), 'companyOfficers': info.get('companyOfficers', [])
        }
        return data
    except Exception as e:
        print(f"Critical error fetching fundamental data for {symbol}: {e}")
        raise ValueError(f"Could not retrieve critical data for {symbol}.")

def fetch_logo(info):
    try:
        if 'logo_url' in info and info['logo_url']:
            response = requests.get(info['logo_url'], stream=True, timeout=5)
            if response.status_code == 200: return ImageReader(BytesIO(response.content))
    except Exception: return None
    return None

def fetch_price(symbol, period="3y"):
    df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True, session=session)
    return df if not df.empty else None

def compute_technical_indicators(df):
    if df is None: return None
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    df['bb_mid'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def fetch_news_yfinance(symbol, limit=5):
    news = []
    try:
        ticker = yf.Ticker(symbol, session=session)
        news_data = ticker.news
        if news_data:
            for item in news_data[:limit]:
                news.append({'title': item.get('title', 'N/A'), 'publisher': item.get('publisher', 'N/A')})
    except Exception as e:
        print(f"Error fetching news from yfinance: {e}")
    return news

def get_peer_comparison(peers):
    peer_details = []
    for peer_symbol in peers:
        try:
            info = yf.Ticker(peer_symbol, session=session).info
            peer_details.append({'name': info.get('shortName', peer_symbol), 'pe': info.get('trailingPE'), 'roe': info.get('returnOnEquity')})
        except Exception: continue
    return pd.DataFrame(peer_details)

def generate_key_highlights(fundamentals, tech_analysis):
    highlights = []
    rec = generate_recommendation(fundamentals, tech_analysis)
    highlights.append(f"<b>Overall Recommendation:</b> Based on a composite analysis, the current recommendation is to <b>{rec}</b>.")
    pe = fundamentals.get('Trailing PE')
    if pe is not None: highlights.append(f"<b>Valuation:</b> The stock's P/E ratio of {pe:.2f} suggests a {'fair' if 15 <= pe <= 30 else 'potentially high' if pe > 30 else 'potentially low'} valuation.")
    if any("Bullish" in s for s in tech_analysis): highlights.append("<b>Technical Trend:</b> The stock is showing bullish long-term trend signals.")
    elif any("Bearish" in s for s in tech_analysis): highlights.append("<b>Technical Trend:</b> The stock is showing bearish long-term trend signals.")
    roe = fundamentals.get('ROE')
    if roe is not None: highlights.append(f"<b>Company Performance:</b> With a Return on Equity of {roe*100:.2f}%, the company shows {'strong' if roe > 0.15 else 'moderate'} profitability.")
    return highlights

def generate_recommendation(fundamentals, tech_analysis):
    score = 0; roe = fundamentals.get('ROE'); pe = fundamentals.get('Trailing PE'); de = fundamentals.get('Debt to Equity')
    if roe is not None and roe > 0.15: score += 1
    if pe is not None and pe < 30: score += 1
    if de is not None and de < 150: score += 1
    if any("Bullish" in s for s in tech_analysis): score += 1
    if any("Overbought" in s for s in tech_analysis): score -= 1
    if score >= 3: return "BUY"
    if score >= 1: return "HOLD"
    return "SELL"

def interpret_technical(df):
    analysis = []
    if df is None or df.empty: return ["Technical data not available."]
    latest = df.iloc[-1]
    sma_50 = latest.get('sma_50'); sma_200 = latest.get('sma_200'); close_price = latest.get('Close')
    bb_upper = latest.get('bb_upper'); macd_line = latest.get('macd_line'); macd_signal = latest.get('macd_signal'); rsi = latest.get('rsi')
    if pd.notna(sma_50) and pd.notna(sma_200): analysis.append("<b>Trend (SMA):</b> " + ("Bullish (Golden Cross)" if sma_50 > sma_200 else "Bearish (Death Cross)"))
    if pd.notna(bb_upper) and pd.notna(close_price): analysis.append("<b>Volatility (Bollinger):</b> " + ("High (Price above upper band)" if close_price > bb_upper else "Normal"))
    if pd.notna(macd_line) and pd.notna(macd_signal): analysis.append("<b>Momentum (MACD):</b> " + ("Positive (MACD above signal)" if macd_line > macd_signal else "Negative (MACD below signal)"))
    if pd.notna(rsi): analysis.append(f"<b>Strength (RSI):</b> {('Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral')} at {rsi:.2f}")
    return analysis

class ReportPDF:
    def __init__(self, filepath, fundamentals):
        self.doc = SimpleDocTemplate(filepath, pagesize=A4, rightMargin=inch*0.5, leftMargin=inch*0.5, topMargin=inch*0.5, bottomMargin=inch*0.5)
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, leading=14))
        self.story = []
        self.fundamentals = fundamentals
        self.navy_blue_header_style = TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#000080')), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f0f0f0")), ('GRID', (0,0), (-1,-1), 1, colors.black)])

    def draw_border(self, canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#000080')); canvas.setLineWidth(2)
        canvas.rect(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)
        canvas.restoreState()

    def create_pdf(self, price_df, news, peer_df, tech_analysis, recommendation, key_highlights):
        self.story.append(Spacer(1, 2*inch)); logo_image = self.fundamentals.get('logo_image')
        if logo_image: logo_image.drawWidth = 1.5*inch; logo_image.drawHeight = 1.5*inch; self.story.append(logo_image)
        self.story.append(Spacer(1, 0.25*inch)); self.story.append(Paragraph("Saketh Equity Research", self.styles['Title']))
        self.story.append(Spacer(1, 0.5*inch)); self.story.append(Paragraph(f"Professional Equity Report For:", ParagraphStyle(name='sub', parent=self.styles['h2'], alignment=TA_CENTER)))
        self.story.append(Paragraph(self.fundamentals.get('Company Name', ''), ParagraphStyle(name='main', parent=self.styles['h1'], alignment=TA_CENTER)))
        self.story.append(Spacer(1, 2*inch)); self.story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
        self.story.append(PageBreak())

        rec_color = {'BUY': colors.green, 'HOLD': colors.orange, 'SELL': colors.red}.get(recommendation, colors.black)
        rec_style = ParagraphStyle(name='Recommendation', parent=self.styles['h1'], alignment=TA_RIGHT, textColor=rec_color)
        header_data = [[Paragraph(f"<b>Live Price:</b> {self.fundamentals.get('Current Price', 'N/A')}<br/><b>Market Cap:</b> {self.fundamentals.get('Market Cap', 'N/A')}", self.styles['Normal']), Paragraph(recommendation, rec_style)]]
        self.story.append(Table(header_data, colWidths=[4*inch, 2.5*inch], style=[('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        self.story.append(Spacer(1, 12))
        
        self.story.append(Paragraph("Key Report Highlights", self.styles['h3']))
        for point in key_highlights: self.story.append(Paragraph(f"• {point}", self.styles['Normal']))
        self.story.append(Spacer(1, 12))
        
        self.story.append(Paragraph("Company Overview", self.styles['h3']))
        self.story.append(Paragraph(self.fundamentals.get('Description', 'N/A') or "No company overview available.", self.styles['Justify']))
        self.story.append(Spacer(1, 12))
        
        self.story.append(Paragraph("Key Managerial Personnel", self.styles['h3']))
        officers = self.fundamentals.get('companyOfficers', [])
        if officers:
            for officer in officers[:5]:
                if 'name' in officer and 'title' in officer: self.story.append(Paragraph(f"• <b>{officer['name']}</b>, <i>{officer['title']}</i>", self.styles['Normal']))
        else: self.story.append(Paragraph("Managerial data could not be retrieved.", self.styles['Normal']))
        self.story.append(PageBreak())

        self.story.append(Paragraph("Financial Summary (Annual, in ₹ Cr)", self.styles['h3']))
        financials = self.fundamentals.get('financials'); balance_sheet = self.fundamentals.get('balance_sheet')
        if financials is not None and not financials.empty and balance_sheet is not None and not balance_sheet.empty:
            fin_data = [['Metric', financials.columns[0].strftime('%Y'), financials.columns[1].strftime('%Y'), financials.columns[2].strftime('%Y')]]
            fin_items = ['Total Revenue', 'Net Income', 'Total Assets', 'Total Liabilities Net Minority Interest']
            all_fins = pd.concat([financials, balance_sheet])
            for item in fin_items:
                if item in all_fins.index: row = [item] + [f"{val/1e7:,.0f}" for val in all_fins.loc[item].iloc[:3]]; fin_data.append(row)
            self.story.append(Table(fin_data, style=self.navy_blue_header_style))
        else: self.story.append(Paragraph("Annual financial data not available.", self.styles['Normal']))
        self.story.append(Spacer(1, 12))
        
        self.story.append(Paragraph("Quarterly Performance (in ₹ Cr)", self.styles['h3']))
        q_financials = self.fundamentals.get('quarterly_financials')
        if q_financials is not None and not q_financials.empty:
            q_data = [['Metric', q_financials.columns[0].strftime('%b %Y'), q_financials.columns[1].strftime('%b %Y'), q_financials.columns[2].strftime('%b %Y'), q_financials.columns[3].strftime('%b %Y')]]
            q_items = ['Total Revenue', 'Net Income']
            for item in q_items:
                if item in q_financials.index: row = [item] + [f"{val/1e7:,.0f}" for val in q_financials.loc[item]]; q_data.append(row)
            self.story.append(Table(q_data, style=self.navy_blue_header_style))
        else: self.story.append(Paragraph("Quarterly performance data not available.", self.styles['Normal']))
        self.story.append(PageBreak())

        self.story.append(Paragraph("Technical Charts & Analysis", self.styles['h3']))
        tech_summary = Paragraph("<br/>".join([f"• {item}" for item in tech_analysis]), self.styles['Normal'])
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [2, 1, 1]})
            ax1.plot(price_df.index, price_df['Close'], label='Close', color='blue'); ax1.plot(price_df.index, price_df['sma_50'], label='50-SMA', linestyle='--'); ax1.plot(price_df.index, price_df['sma_200'], label='200-SMA', linestyle='--')
            ax1.fill_between(price_df.index, price_df['bb_upper'], price_df['bb_lower'], color='gray', alpha=0.1)
            ax1.set_ylabel('Price (₹)'); ax1.legend(); ax1.grid(True)
            ax2.plot(price_df.index, price_df['macd_line'], label='MACD'); ax2.plot(price_df.index, price_df['macd_signal'], label='Signal', linestyle='--')
            ax2.bar(price_df.index, price_df['macd_hist'], color='gray', alpha=0.5)
            ax2.set_ylabel('MACD'); ax2.legend(); ax2.grid(True)
            ax3.plot(price_df.index, price_df['rsi'], label='RSI', color='purple'); ax3.axhline(70, color='r', linestyle='--'); ax3.axhline(30, color='g', linestyle='--')
            ax3.set_ylabel('RSI'); ax3.legend(); ax3.grid(True)
            # --- BUG FIX for chart labels ---
            for ax in [ax1, ax2, ax3]: plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
            plt.tight_layout()
            img_buffer = BytesIO(); fig.savefig(img_buffer, format='png', dpi=300); plt.close(fig)
            chart_image = Image(img_buffer, width=4*inch, height=4.5*inch)
            t_data = [[chart_image, tech_summary]]
            self.story.append(Table(t_data, colWidths=[4.2*inch, 2.3*inch], style=[('VALIGN', (0,0), (-1,-1), 'TOP')]))
        except Exception as e:
            self.story.append(Paragraph(f"<b>Could not generate charts due to a technical error.</b>", self.styles['Normal']))
        
        self.story.append(Spacer(1, 12))
        self.story.append(Paragraph("Peer Comparison", self.styles['h3']))
        if not peer_df.empty:
            peer_data = [['Company', 'P/E Ratio', 'ROE']]
            for _, row in peer_df.iterrows(): peer_data.append([row['name'], f"{row['pe']:.2f}" if pd.notna(row['pe']) else 'N/A', f"{row['roe']*100:.2f}%" if pd.notna(row['roe']) else 'N/A'])
            self.story.append(Table(peer_data, style=self.navy_blue_header_style))
        else: self.story.append(Paragraph("Peer comparison data could not be retrieved.", self.styles['Normal']))
        self.story.append(PageBreak())

        self.story.append(Paragraph("Recent News", self.styles['h3']))
        if news:
            for n in news: self.story.append(Paragraph(f"• <b>{n['title']}</b> <i>({n['publisher']})</i>", self.styles['Normal']))
        else: self.story.append(Paragraph("Recent news could not be fetched at this time.", self.styles['Normal']))
        
        self.story.append(Spacer(1, 24))
        self.story.append(Paragraph("Disclaimer", self.styles['h3']))
        disclaimer_text = "This report is for informational and educational purposes only and does not constitute a recommendation to buy or sell any security. The information contained herein has been obtained from sources believed to be reliable, but its accuracy and completeness are not guaranteed. Saketh Equity Research is not a registered investment advisor. All investment decisions should be made with the help of a qualified financial professional. Past performance is not indicative of future results."
        self.story.append(Paragraph(disclaimer_text, self.styles['Justify']))

    def generate(self):
        self.doc.build(self.story, onFirstPage=self.draw_border, onLaterPages=self.draw_border)

# ======================================================================
# MAIN WORKFLOW AND FLASK ROUTES
# ======================================================================

def create_report(stock_ticker):
    symbol = stock_ticker
    fundamentals = fetch_company_info(symbol) # This now raises ValueError on failure
    
    price_df = fetch_price(symbol)
    price_df_tech = compute_technical_indicators(price_df)
    
    tech_analysis = interpret_technical(price_df_tech)
    recommendation = generate_recommendation(fundamentals, tech_analysis)
    key_highlights = generate_key_highlights(fundamentals, tech_analysis)
    
    peers = PEER_MAPPING.get(symbol, [])
    peer_df = get_peer_comparison(peers)
    news = fetch_news_yfinance(symbol)

    filename = f"{symbol.replace('.', '_')}_Pro_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    pdf = ReportPDF(filepath, fundamentals)
    pdf.create_pdf(price_df_tech, news, peer_df, tech_analysis, recommendation, key_highlights)
    
    return filename

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/stocks')
def get_stocks(): return jsonify(SECTOR_STOCK_MAPPING)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        stock_ticker = request.form['stock_select']
        if not stock_ticker: return jsonify({'error': 'No stock selected.'}), 400
        pdf_filename = create_report(stock_ticker)
        return jsonify({'download_url': f"/download/{pdf_filename}"})
    except ValueError:
        return jsonify({'error': f"The data source is currently busy or unavailable for {stock_ticker}. This can happen with free APIs. Please wait 30 seconds and try again."}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': 'An internal server error occurred. Please check the logs.'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
