### instate: predict the state of residence from last name 

Using the Indian electoral rolls data (2017), we provide a Python package that takes the last name of a person and gives its distribution across states. The underlying data for the package can be accessed at: https://doi.org/10.7910/DVN/ENXOJE

## Potential Use Cases

India has 22 official languages. And to serve such a diverse language base is a challenge for businesses and surveyors. To the extent that businesses have access to the last name (and no other information) and in absence of other data that allows us to model a person's spoken language, the distribution of last name across states is the best we have.

### Installation

## API

instate exposes only 1 function `last_state` that takes a pandas dataframe, the column name with the last names, and produces a dataframe with XX more columns, reflecting the number of states for which we have the data. 

```
import pandas as pd
last_dat <- pd.read_csv("last_dat.csv")
last_state_dat <- last_state(last_dat, "last_name")
```


### Authors
Atul Dhingra, Gaurav Sood

### License
