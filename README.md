# a2

## Browser-Use Agent - Accumulator Bet Finder with Stealth Mode

This repository contains a GitHub Actions workflow that runs a browser-use agent every hour to find top accumulator bet matches with the least risk occurring in the next three hours. The agent uses OLBG as its primary aggregator source and operates in stealth mode with human-like behavioral patterns.

### Features

- **Automated Hourly Monitoring**: GitHub Actions workflow runs every hour using cron schedule
- **AI-Powered Browser Automation**: Uses the browser-use agent (https://github.com/browser-use/browser-use) with Gemini LLM orchestration
- **OLBG Integration**: Uses OLBG (Online Betting Guide) as the primary source for betting tips and match analysis
- **Smart Bet Finding**: Searches for low-risk accumulator betting opportunities with a 3-hour time window
- **Stealth Mode**: Military-grade human behavioral emulation system to defeat bot detection
  - Realistic mouse movements with Fitts's Law modeling
  - Human-like typing patterns with psychological coherence
  - Adaptive timing delays based on cognitive state
  - Emotional state modeling (calm, focused, stressed, fatigued, alert)
- **Result Persistence**: Automatically commits search results as JSON files to the repository

### Setup

#### Prerequisites

1. **Gemini API Key**: The workflow uses Google's Gemini 2.5 Flash Preview model for the LLM orchestration
2. **GitHub Repository Secret**: Add your Gemini API key as a repository secret named `GEMINI_API_KEY`

#### Adding the API Key

1. Go to your repository settings
2. Navigate to Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `GEMINI_API_KEY`
5. Value: Your Gemini API key (from https://key.ematthew477.workers.dev)
6. Click "Add secret"

### Usage

#### Automatic Execution

The workflow runs automatically every hour (at the top of the hour) via GitHub Actions.

#### Manual Execution

You can also trigger the workflow manually:

1. Go to the "Actions" tab in your repository
2. Select "Browser Agent - Accumulator Bet Finder" workflow
3. Click "Run workflow"
4. Select the branch and click "Run workflow"

### Results

Search results are saved as JSON files with the naming pattern:
- `accumulator_bets_YYYYMMDD_HHMMSS.json` - Successful search results
- `accumulator_bets_YYYYMMDD_HHMMSS_error.json` - Error logs if the search fails

Each result file contains:
- Timestamp of the search
- Search type ("accumulator_bets")
- Primary source ("OLBG")
- Time window (current time + 3 hours)
- Agent execution history with discovered matches
- Stealth mode status and psychological state
- Match details including teams, odds, risk assessments, and OLBG tipster ratings
- Status and any error information

### Components

- `browser_agent.py` - Python script that runs the browser-use agent with stealth mode
- `stealth.py` - Military-grade human behavioral emulation system
- `requirements.txt` - Python dependencies including numpy and scipy for behavioral modeling
- `.github/workflows/browser-agent.yml` - GitHub Actions workflow configuration

### Technical Details

- **Python Version**: 3.11
- **Browser**: Chromium (installed via Playwright)
- **LLM**: Google Gemini 2.5 Flash Preview (via native browser-use ChatGoogle integration)
- **API Endpoint**: Custom Gemini mirror at https://key.ematthew477.workers.dev
- **Framework**: browser-use with native Gemini support
- **Stealth System**: Computational psychology & behavioral biometrics (Agent: Silus)
- **Primary Source**: OLBG (olbg.com) for betting tips and match aggregation