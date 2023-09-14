# bts-shiny
Shiny for Python App to improve picks for MLB's Beat the Streak contest
## Database
Daily data fetching from Baseball Savant and MLB Stats API and saving to MongoDB:<br>
[![daily](https://github.com/peteb206/bts-shiny/actions/workflows/daily.yml/badge.svg)](https://github.com/peteb206/bts-shiny/actions/workflows/daily.yml)
## Predictive Model
I am currently using a Logistic Regression classifier to calculate the probability of each player getting a hit on a given day.<br><br>
The model uses MLB data from 2016 to 2023 and takes in 36 features (all numeric), normalizes them to a -1 to 1 scale, reduces them to X principal components (X is determined how many components it takes to explain 99% of the variance).<br><br>
When applied to test data (game dates reserved for testing), the model was able to achieve a pick accuracy of 79% and a max streak of 38 when doubling down every day. See [log_reg.ipynb](models/log_reg.ipynb) and supporting code to see how this model was trained.
## App Deployment to shinyapps.io
### General Instructions
https://shiny.posit.co/py/docs/deploy-cloud.html#shinyapps.io
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