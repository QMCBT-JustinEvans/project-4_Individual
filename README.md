# <h><i><u><font size="20">Predicting Life Expectancy</font></u></i></h>

* by Justin Evans
    * GitHub: https://github.com/QMCBT-JustinEvans/project-4_Individual/blob/main/01_wrangle.ipynb
    * LinkedIn: www.linkedin.com/in/qmcbt
    * E-mail: Justin.Ellis.Evans@gmail.com

# Project Overview:
This is an individual project to demonstrate my ability to source and acquire data independently outside the classroom environment. This individual project is an opportunity for me to showcase my ability to plan and execute a project that follows the full Data Science pipeline from start to finish with a presentable deliverable that has meaning and value in its content. I have chosen to use data from the [World Health Organization (WHO)](https://www.who.int/) because it is trustworthy and contains global input that spans many years.

# Goals: 
* Locate, Acquire, and Prepare data from a reputable source 
* Explore, Model, and Evaluate to select a model that will outperform Baseline
* Provide valid and insightful Observations and Recommendations 

# Reproduction of this Data:
* This project can be reproduced by cloning my GitHub repository [here:](https://github.com/QMCBT-JustinEvans/project-4_Individual) https://github.com/QMCBT-JustinEvans/project-4_Individual or by simply downloading at a minimum, the below listed files into a local folder and running the Final_Project.ipynb
    * 04_Final_Project.ipynb
    * QMCBT_wrangle.py
    * QMCBT_explore.py
    * QMCBT_model.py
    * leam2.csv
    
    
* The below files can be used to walk step by step through the Data Science Process used to create the 04_Final_Project.ipynb file.
    * 01_wrangle2.ipynb
    * 02_explore.ipynb
    * 03_model.ipynb
    
    
* Data for this project was acquired from the [World Health Organization (WHO)](https://www.who.int/) using the the [Global Health Obsevatory (GHO)](https://www.who.int/data/gho) data webservice, [Athena](https://www.who.int/data/gho/info/athena-api-examples).
    * World Health Organization (WHO) - [https://www.who.int/](https://www.who.int/)
    * Global Health Obsevatory (GHO) - [https://www.who.int/data/gho](https://www.who.int/data/gho)
    * Athena data webservice - [https://www.who.int/data/gho/info/athena-api-examples](https://www.who.int/data/gho/info/athena-api-examples)
    
# Initial Thoughts
I started with an enormous data set that I thought would be incredible for predictions but after cleaning the data and preparing it for Machine Learning, there were so many NaN entries that the data was unusable.

# The Plan
* Follow the Data Science process map
* Acquire data from WHO data webservice
* Prepare data
* Explore data in search of correlation to life expectancy
* Answer the following initial question:

    * **Question 1.** Is there a relationship between Gender and life_expectancy?

    * **Question 2.** Is the Life Expectancy of Females greater than the Life Expectancy of Males? 

    * **Question 3.** Is there a relationship between Year and life_expectancy?

    * **Question 4.** Is there a relational difference between the four observable Years in our data?


* Develop a Model to predict life expectancy.
    * Use drivers identified to explore and build predictive models.
    * Evaluate models on train and validate data using MSE (Mean square Error)
    * Select the best model based on the least MSE
    * Evaluate the best model on test data
* Draw conclusions


   
# Data Dictionary
* Documentation is hyperlinked

|[GHO Code](https://apps.who.int/gho/athena/api/GHO)      |[Documentation](https://www.who.int/data/gho/indicator-metadata-registry)|[Global Health Observatory](https://www.who.int/data/gho) (GHO) Code Description|
|:-------------|:-----------:|:------------------------------------------------|
|WHOSIS_000001|[📖](https://www.who.int/data/gho/indicator-metadata-registry/imr-details/65)|Life expectancy at birth (years)| 
|WHOSIS_000002|[📖](https://www.who.int/data/gho/indicator-metadata-registry/imr-details/66)|Healthy life expectancy (HALE) at birth (years)| 
|WHOSIS_000007|[📖](https://www.who.int/data/gho/indicator-metadata-registry/imr-details/3443)|Healthy life expectancy (HALE) at age 60 (years)| 
|WHOSIS_000015|[📖](https://www.who.int/data/gho/indicator-metadata-registry/imr-details/2977)|Life expectancy at age 60 (years)| 
|YEAR         |    |The Year that the data was collected| 
|SEX_BTSX     |    |Both Sex; this is a combination of both Male and Female Gender (1=True and 0=False)|
|SEX_MLE      |    |Male Gender (1=True and 0=False)|
|SEX_FMLE     |    |Female Gender (1=True and 0=False)|
|life_expectancy|  |This feature is made from the mean of each of the WHOSIS features|


# Takeaways and Conclusions

## Exploration Summary of Findings:
* Gender seems to have an impact on life expectancy
* Both Male and Female life expectancies raise over the years
* In general year over year the life expectancy rate seems to maintain an upward trend
* Women have a higher life expectany than men

## Modeling Summary:
* All models did slightly better than baseline.
* None of the models were within acceptable proximity to actual target results

* Our top model ```Simple Linear Regression Model```  was run on test data and performed better than baseline as expected and even outperformed its previous score on validation by approximately three base points.

**For this itteration of modeling we have a model that beats baseline.**    

# Conclusions: 
* **Exploration:** 
    * We asked 4 Questions using T-Test and Anova Statistical testing to afirm our hypothesis
    * In general year over year the life expectancy rate seems to maintain an upward trend
    * Women have a higher life expectany than men
* **Modeling:**
    * We trained and evaluated 6 different Linear Regression Models, all of which outperformed baseline 
    * We chose the Simple Linear Regression Model as our best performing model
    * When evaluated on Test, it continued to outperform baseline and surpased its previous performance on validate
* **Recommendations:**
    * I think we should hold off on deploying this model.
    * Even though it beat baseline, it came nowhere near actual.
    * We can acquire a much better dataset given more time
* **Next Steps:**
    * I would like to request more time to investigate the data available on the Athena data webservice managed by the World Health Organization.
    * I also came across some similar projects that I can reference to research their findings in comparison to my own.