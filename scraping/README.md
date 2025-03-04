# Phase 1: Data Collection

This directory contains scripts for scraping NBA player and game data.

## Objectives
- Collect comprehensive historical NBA player data
- Gather related contextual information that may affect performance
- Ensure data quality and consistency
- Create a robust pipeline for regular data updates

## Data Sources
- [Basketball-Reference](https://www.basketball-reference.com/) - Historical player and team statistics
- [NBA.com](https://www.nba.com/stats/) - Official NBA statistics
- [ESPN](https://www.espn.com/nba/stats) - Player statistics and game information
- [Odds API](https://the-odds-api.com/) or [Sportsbook Review](https://www.sportsbookreview.com/) - Historical betting lines

## Implementation Steps

### 1. Player Game Logs Scraper
- Create a script to scrape player game logs (e.g., `player_scraper.py`)
- Extract per-game statistics:
  - Minutes played
  - Points scored
  - Field goal attempts/percentage
  - Three-point attempts/percentage
  - Free throw attempts/percentage
  - Other relevant stats (rebounds, assists, etc.)
- Store data with appropriate player and game identifiers

### 2. Team and Context Data Scraper
- Create a script to gather team-level data (e.g., `team_scraper.py`)
- Extract information such as:
  - Team offensive/defensive ratings
  - Pace of play
  - Rest days between games
  - Home/away status
  - Back-to-back game indicators

### 3. Player Status Scraper
- Create a script to collect player status information (e.g., `player_status_scraper.py`)
- Track:
  - Injury reports
  - Starting lineup changes
  - Minutes restrictions
  - Recent role changes

### 4. Historical Betting Lines Scraper
- Create a script to gather historical betting lines (e.g., `odds_scraper.py`)
- Focus on player points over/under lines
- Store odds from multiple sportsbooks if available

## Data Storage
- Store raw scraped data in the `data/raw/` directory
- Use consistent file naming conventions:
  - `player_data_YYYY_MM_DD.csv`
  - `team_data_YYYY_MM_DD.csv`
  - `player_status_YYYY_MM_DD.csv`
  - `betting_lines_YYYY_MM_DD.csv`
- Implement data validation checks to ensure integrity

## Technical Requirements
- Use `requests` and `BeautifulSoup` or `Selenium` for scraping
- Implement proper rate limiting to avoid IP blocks
- Add logging for tracking scraping progress and errors
- Include error handling and retry logic
- Document any site-specific scraping challenges

## Deliverables
- Scraping scripts for all data sources
- Raw data files saved in specified formats
- Documentation on how to run the scripts
- Log of data collection activities

## Tips
- Consider using rotating proxies for large-scale scraping
- Respect website terms of service
- Add random delays between requests
- Create a unified scraper interface for consistent operation