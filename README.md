# bts-shiny
Shiny for Python App to improve picks for MLB's Beat the Streak contest

## Database
MongoDB

Daily data fetching from Baseball Savant and MLB Stats API<br>
[![daily](https://github.com/peteb206/bts-shiny/actions/workflows/daily.yml/badge.svg)](https://github.com/peteb206/bts-shiny/actions/workflows/daily.yml)

## Deployment
### Non-git file requirements 
- \<path\>/bts-shiny/rsconnect-python/bts-shiny.json:
```
{
    "https://api.shinyapps.io": {
        "server_url": "https://api.shinyapps.io",
        "filename": "/Users/peteberryman/Desktop/bts-shiny",
        "app_url": "https://peteberryman.shinyapps.io/bts-shiny/",
        "app_id": 9742736,
        "app_guid": null,
        "title": "bts-shiny",
        "app_mode": "python-shiny",
        "app_store_version": 1
    }
}
```
- \<path\>/bts-shiny/MongoDB.txt:
```
<password>
```
### Command
```
rsconnect deploy shiny ./
```