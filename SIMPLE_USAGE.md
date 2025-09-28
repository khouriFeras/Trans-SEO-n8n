# 🎯 Simple Usage Guide

## ✅ **Working Solutions (No N8N Required)**

Since N8N Docker doesn't have Python, here are your **proven working options**:

### **Option 1: One-Command Solution** ⭐ (Easiest)

```bash
cd N8Nscripts
python auto_translate_from_sheets.py "YOUR_GOOGLE_SHEETS_ID" "output.xlsx" "tools and equipment" 5
```

**What this does:**
1. Downloads CSV from your Google Sheets automatically
2. Translates Title + Body(HTML) to Arabic
3. Generates SEO meta_title, meta_description, product_title_ar
4. Saves as Excel file

### **Option 2: Two-Step Process**

1. **Download from Google Sheets:**
   - Go to your Google Sheets → File → Download → CSV
   - Save as `my_data.csv`

2. **Translate:**
   ```bash
   cd N8Nscripts  
   python n8n_translator.py "my_data.csv" "translated.xlsx" "tools and equipment" 5
   ```

### **Option 3: Direct Excel File**

```bash
cd N8Nscripts
python n8n_translator.py "test tans.xlsx" "output.xlsx" "tools and equipment" 5
```

## 🔧 **How to Get Your Google Sheets ID**

From this URL: `https://docs.google.com/spreadsheets/d/1Y8QzQzQzQzQzQzQzQzQzQzQzQ/edit`

The ID is: `1Y8QzQzQzQzQzQzQzQzQzQzQzQzQzQzQzQzQzQ`

## 📊 **What Gets Translated**

**Input columns:**
- `Title` → Product name
- `Body (HTML)` → Product description

**Generated columns:**
- `meta_title` → Arabic SEO title (50-60 chars)
- `meta_description` → Arabic SEO description (150-160 chars)  
- `product_title_ar` → Arabic product name

**All other columns preserved** (Tags, SKU, Price, Vendor, etc.)

## ⚡ **Quick Test**

```bash
cd N8Nscripts
python n8n_translator.py "test tans.xlsx" "quick_test.xlsx" "tools and equipment" 2
```

This will:
- Process 2 rows from your test file
- Generate Arabic translations
- Create `quick_test.xlsx` with results
- Take ~10-15 seconds

## 🎉 **Perfect for Your Use Case**

- ✅ **Works with your exact columns** (Title, Body HTML, etc.)
- ✅ **Generates Arabic SEO content** 
- ✅ **No N8N Docker issues**
- ✅ **Uses your OpenAI API key from .env**
- ✅ **Tested and verified working**

**This is your complete, working solution!** 🚀

