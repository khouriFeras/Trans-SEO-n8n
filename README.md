# Universal Translation and SEO Generator for N8N

A comprehensive solution for translating product data from English to Arabic and generating SEO-optimized content. Designed for N8N automation workflows with Google Sheets integration and file download capabilities.

## 🚀 Features

- **Universal Product Support**: Works with any product type (tools, car parts, kitchen appliances, etc.)
- **Google Sheets Integration**: Direct integration with Google Sheets via N8N
- **File Download**: Automatic Excel file generation and download in N8N
- **SEO Optimization**: Generates Arabic meta titles, descriptions, and product titles
- **Web Service**: RESTful API for N8N HTTP requests
- **Excel/CSV Support**: Processes both Excel and CSV files
- **Async Processing**: Fast concurrent processing with configurable workers
- **Error Handling**: Robust error handling with retries and validation

## 📁 Files Overview

| File                             | Purpose                                            |
| -------------------------------- | -------------------------------------------------- |
| seo-trans.json                   | **Main N8N workflow** - Import this into N8N |
| `web_service.py`               | Flask web service for translation API              |
| `universal_translation_seo.py` | Core translation engine with all functionality     |
| `requirements.txt`             | Python dependencies                                |
| `env_example.txt`              | Environment variables template                     |
| `SIMPLE_USAGE.md`              | Quick usage guide                                  |
| `README.md`                    | This documentation file                            |

## 🛠️ Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Copy `env_example.txt` to `.env` and add your API key:

```bash
cp env_example.txt .env
```

Edit `.env`:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the Web Service

```bash
python web_service.py
```

The service will run on `http://localhost:5000`

### 4. Import N8N Workflow

1. Open N8N
2. Import seo-trans.json
3. Replace `YOUR_GOOGLE_SHEETS_ID_HERE` with your actual Google Sheets ID
4. Run the workflow!

## 🔧 How It Works

### N8N Workflow Flow

1. **Manual Trigger** → Start the workflow
2. **Google Sheets** → Read product data from your sheet
3. **Prepare CSV** → Convert Google Sheets data to CSV format
4. **Call Translation** → Send data to web service for processing
5. **Process Results** → Extract file ID and prepare download URL
6. **Download File** → Get the translated Excel file
7. **Final Results** → Display success information and file details

### Web Service Endpoints

| Endpoint                | Method | Purpose                             |
| ----------------------- | ------ | ----------------------------------- |
| `/health`             | GET    | Check service status                |
| `/test`               | GET    | Test endpoint with service info     |
| `/process`            | POST   | Process CSV data and return file ID |
| `/download/<file_id>` | GET    | Download translated Excel file      |

## 📊 Supported Data Format

The system works with Google Sheets containing these columns:

| Column          | Required | Description                    |
| --------------- | -------- | ------------------------------ |
| `Title`       | ✅       | Product name in English        |
| `Body (HTML)` | ✅       | Product description in English |
| `Brand`       | ❌       | Brand name (optional but recommanded )|
| `Category`    | ❌       | Product category (optional)    |
| `SKU`         | ❌       | Product SKU (optional)         |

## 📤 Generated Content

The system adds these new columns to your Excel file:

| Column                 | Content                                    | Length      |
| ---------------------- | ------------------------------------------ | ----------- |
| `arabic description` | Full Arabic product description with specs | Variable    |
| `meta_title`         | SEO-optimized Arabic meta title            | ≤60 chars  |
| `meta_description`   | SEO-optimized Arabic meta description      | ≤155 chars |
| `product_title_ar`   | Arabic product title for display           | 50-80 chars |

## 🔄 N8N Integration

### Workflow Configuration

1. **Import** seo-trans.json into N8N
2. **Configure Google Sheets**:

   - Set up OAuth2 authentication
   - Replace `YOUR_GOOGLE_SHEETS_ID_HERE` with your sheet ID
   - Select the correct sheet name (default: "Sheet1")

3. **Run the workflow**:

   - Click "Execute Workflow"
   - The system will process your Google Sheets data
   - Download the translated Excel file from the results

### Customization Options

You can modify the workflow to:

- Change the number of sample rows processed
- Adjust product context (currently "tools and equipment")
- Modify column mappings
- Add error handling nodes

## ⚙️ Advanced Configuration

### Web Service Settings

Edit `web_service.py` to customize:

- Port number (default: 5000)
- Host binding (default: 0.0.0.0)
- File cleanup intervals
- Error handling

### Translation Settings

Edit `universal_translation_seo.py` to customize:

- OpenAI model (default: gpt-4o-mini)
- Processing workers (default: 5)
- Description character limits
- Custom prompts for different product types

## 🛡️ Error Handling

The system includes comprehensive error handling:

- **API Retries**: Automatic retries for OpenAI API failures
- **File Validation**: Input validation and column checking
- **Error Logging**: Detailed logs for troubleshooting
- **Graceful Degradation**: Continues processing even if some items fail

## 📈 Performance Tips

1. **API Key**: Store in `.env` file for security
2. **File Size**: The workflow processes 5 rows by default for testing
3. **Workers**: Default 5 workers for optimal performance
4. **Testing**: Start with small samples to verify setup

## 🔍 Troubleshooting

### Common Issues

**Web Service Not Starting**

```
Connection refused
```

Solution: Make sure `python web_service.py` is running

**Google Sheets Connection Failed**

```
Authentication error
```

Solution: Set up OAuth2 authentication in N8N Google Sheets node

**Translation Failed**

```
API key missing
```

Solution: Check your `.env` file has the correct `OPENAI_API_KEY`

**File Download Failed**

```
File ID not found
```

Solution: Check that the web service is running and the file ID is valid

## Security Notes

- Never hardcode API keys in files
- Use `.env` file for sensitive data
- The web service runs locally by default
- Files are temporarily stored and cleaned up automatically

## Example Usage

### Basic Workflow

1. **Prepare Google Sheets** with your product data
2. **Import N8N workflow** (seo-trans.json)
3. **Configure Google Sheets** connection
4. **Run workflow** - get translated Excel file!

### Custom Product Context

Edit the "Prepare CSV" node in N8N to change the product context:

```javascript
// Change this line in the function:
product_context: 'your_product_type_here'
```

## Getting Started Checklist

- [ ] Install Python dependencies: `pip install -r requirements.txt`
- [ ] Set up `.env` file with your OpenAI API key
- [ ] Start web service: `python web_service.py`
- [ ] Import seo-trans.json into N8N
- [ ] Configure Google Sheets connection in N8N
- [ ] Replace `YOUR_GOOGLE_SHEETS_ID_HERE` with your sheet ID
- [ ] Test with a small sample of data
- [ ] Run the complete workflow!

## Support

For issues or questions:

1. Check the web service console for errors
2. Verify your Google Sheets has the required columns
3. Test with a small sample first
4. Check your OpenAI API key and credits
5. Review N8N execution logs for detailed error information

*Your products are now ready for Arabic-speaking markets with SEO-optimized content!*

---

**Happy translating! 🌟**
