# Customer churn and survival analysis 

### 1. Model comparison 
After fitting Weibull, LogNormal and LogLogistic AFT models to predict customer tenure, the LogNormal one had the lowest
AIC score. I selected it as the final model 

### 2. Significant Features
I guess something may be done incorrectly here, but no covariates were actually statistically significant. However, with 
I think the model can still predict survival and CLV. 

### 3. CLV by segment
I decided to segment by the customer category, region and gender. 
The outputs for each segmented analysis was as follows:

**Customer Category (custcat):**
- Basic service    12974.690068 
- E-service        69671.766454
- Plus service     97983.218778 
- Total service    53204.442645

**Region:**
- Zone 1    47661.288571
- Zone 2    52653.925257
- Zone 3    74785.546116

**Gender:**
- Female    62578.614898
- Male      54464.658213

After analysing the given numbers, we can identify that the most valuable customer segments are _Female Plus service customers_ 
in _Zone 3_. Valuabe here is defined as a combination of highly predicted tenure and high revenue contribution. Even though
no significant covariate was statistically significant (at least according to my codes and outputs:)), segment-level differnces
are clear and can be acted upon later. 

### 4. Retention strategy
To plan retention efforts, I estimated which customers are at risk of churning within the next year using the model’s predicted 
survival probabilities at 12 months. By combining this with their CLV, I calculated an estimated annual retention budget
of approximately 13,031. This means that the sum of expected losses from at-risk customers provides a reasonable ceiling 
for retention investment. Retention strategies should therefore focus on high-CLV, high-risk segments, offering personalized 
promotions, loyalty rewards, and targeted communication to encourage continued subscription. Additionally, upselling 
lower-tier customers to higher-value plans, such as encouraging E-service customers to upgrade to Plus service, can increase 
lifetime value. Continuous monitoring of survival probabilities and updating predictions will allow for dynamic adjustments 
in retention campaigns and budgets, maximizing the long-term value of the customer base.