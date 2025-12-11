# a2

## Browser-Use Agent - Bing Search Monitor

This repository contains a GitHub Actions workflow that runs a browser-use agent every hour to search Bing for "mstchrd" occurring in the next three hours and saves the results to the repository.

### Features

- **Automated Hourly Monitoring**: GitHub Actions workflow runs every hour using cron schedule
- **AI-Powered Browser Automation**: Uses the browser-use agent (https://github.com/browser-use/browser-use) with LLM orchestration
- **Bing Search**: Searches for "mstchrd" with a 3-hour time window
- **Result Persistence**: Automatically commits search results as JSON files to the repository

### Setup

#### Prerequisites

1. **OpenAI API Key**: The workflow uses OpenAI's GPT-4 model for the LLM orchestration
2. **GitHub Repository Secret**: Add your OpenAI API key as a repository secret named `OPENAI_API_KEY`

#### Adding the API Key

1. Go to your repository settings
2. Navigate to Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `OPENAI_API_KEY`
5. Value: Your OpenAI API key
6. Click "Add secret"

### Usage

#### Automatic Execution

The workflow runs automatically every hour (at the top of the hour) via GitHub Actions.

#### Manual Execution

You can also trigger the workflow manually:

1. Go to the "Actions" tab in your repository
2. Select "Browser Agent - Bing Search Monitor" workflow
3. Click "Run workflow"
4. Select the branch and click "Run workflow"

### Results

Search results are saved as JSON files with the naming pattern:
- `results_YYYYMMDD_HHMMSS.json` - Successful search results
- `results_YYYYMMDD_HHMMSS_error.json` - Error logs if the search fails

Each result file contains:
- Timestamp of the search
- Search term used ("mstchrd")
- Time window (current time + 3 hours)
- Agent execution history
- Status and any error information

### Components

- `browser_agent.py` - Python script that runs the browser-use agent
- `requirements.txt` - Python dependencies
- `.github/workflows/browser-agent.yml` - GitHub Actions workflow configuration

### Technical Details

- **Python Version**: 3.11
- **Browser**: Chromium (installed via Playwright)
- **LLM**: OpenAI GPT-4o
- **Framework**: browser-use with langchain-openai integration