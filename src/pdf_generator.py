from fpdf import FPDF
import os
from datetime import datetime

class ECGReport(FPDF):
    def header(self):
        # Professional Blue Header
        self.set_fill_color(15, 23, 42) # Slate
        self.rect(0, 0, 210, 35, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 18)
        self.set_y(10)
        self.cell(0, 10, 'BIO-SIGNAL PRO | CLINICAL DIAGNOSTICS', 0, 1, 'C')
        self.set_font('Helvetica', '', 9)
        self.cell(0, 5, f'LABORATORY RECORD: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | CONFIDENTIAL MEDICAL DATA', 0, 0, 'C')

def sanitize_text(text):
    if not text: return ""
    replacements = {"\u2014": "-", "\u2013": "-", "\u2019": "'", "\u2018": "'", 
                    "\u201d": '"', "\u201c": '"', "\u2022": "*", "\u2265": ">=", "\u2264": "<="}
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode('latin-1', 'replace').decode('latin-1')

def write_formatted_ai_content(pdf, text):
    """
    Parses Markdown-like headers and bold text for a professional look.
    """
    pdf.set_text_color(0, 0, 0)
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(2)
            continue
            
        # Detect Headers (### Section)
        if line.startswith('###'):
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(16, 185, 129) # Emerald Green
            clean_header = line.replace('###', '').strip()
            pdf.cell(0, 10, clean_header, 0, 1, 'L')
            pdf.set_draw_color(16, 185, 129)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x()+50, pdf.get_y())
            pdf.ln(2)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', '', 10)
        
        # Detect Bullet Points
        elif line.startswith('*') or line.startswith('-'):
            pdf.set_x(15)
            # Handle Bold within bullet
            parts = line.split('**')
            pdf.set_font('Helvetica', 'B', 10)
            pdf.write(7, "- ")
            for i, part in enumerate(parts):
                style = 'B' if i % 2 == 1 else ''
                pdf.set_font('Helvetica', style, 10)
                pdf.write(7, part)
            pdf.ln(7)
            
        # Normal Text with Bold Detection
        else:
            pdf.set_x(10)
            parts = line.split('**')
            for i, part in enumerate(parts):
                style = 'B' if i % 2 == 1 else ''
                pdf.set_font('Helvetica', style, 10)
                pdf.write(6, part)
            pdf.ln(6)

def create_pdf_report(metrics, stats, ai_insight, plot_paths, output_path):
    pdf = ECGReport()
    pdf.add_page()
    pdf.set_margins(10, 40, 10)
    
    # 1. CORE VITALS
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, 'I. PRIMARY CARDIAC VITALS', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 11)
    
    col1_v = [f"Heart Rate: {metrics.get('HR')} BPM", f"SDNN: {metrics.get('SDNN')} ms"]
    col2_v = [f"Mean RR: {metrics.get('Mean_RR')} ms", f"RMSSD: {metrics.get('RMSSD')} ms"]
    
    curr_y = pdf.get_y()
    for line in col1_v: pdf.cell(95, 7, f"  - {line}", 0, 1, 'L')
    pdf.set_y(curr_y)
    for line in col2_v: 
        pdf.set_x(105)
        pdf.cell(95, 7, f"  - {line}", 0, 1, 'L')
    pdf.ln(10)

    # 2. GRAPHICAL SECTION
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'II. DIAGNOSTIC WAVEFORMS', 0, 1, 'L')
    if plot_paths.get('ecg'):
        pdf.image(plot_paths['ecg'], x=10, w=190)
        pdf.ln(5)
    
    if pdf.get_y() > 180: pdf.add_page()

    # 3. STATS TABLE
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'III. EXTENDED HRV STATISTICS', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_fill_color(243, 244, 246)
    
    headers = ['METRIC', 'VALUE', 'METRIC', 'VALUE']
    cw = pdf.epw / 4
    for h in headers: pdf.cell(cw, 8, h, 1, 0, 'C', True)
    pdf.ln()
    
    rows = [
        ['Variance', f"{stats.get('Variance',0):.1f}", 'Skewness', f"{stats.get('Skewness',0):.3f}"],
        ['Kurtosis', f"{stats.get('Kurtosis',0):.3f}", 'pNN50', f"{stats.get('pNN50',0):.1f}%"],
        ['Range', f"{stats.get('Range',0):.1f} ms", 'Median RR', f"{stats.get('Median_RR',0):.1f} ms"]
    ]
    for row in rows:
        for item in row: pdf.cell(cw, 8, item, 1, 0, 'C')
        pdf.ln()
    
    # 4. AI INSIGHTS PAGE
    if ai_insight:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(16, 185, 129)
        pdf.cell(0, 15, 'IV. GEMINI AI CLINICAL CONSULTATION', 0, 1, 'C')
        pdf.ln(5)
        write_formatted_ai_content(pdf, sanitize_text(ai_insight))

    pdf.output(output_path)
    return output_path
