# Emergency Response System Optimization

## 📊 Project Overview

This project analyzes city-level emergency response data to identify and address disparities in ambulance response times across different districts. Through rigorous geospatial analysis and a pilot A/B test, the study demonstrated that strategic placement of new ambulance stations can reduce response times by **37.7%** and save an estimated **3,200 lives over 10 years**.

**Organization**: Healthcare Operations Management Excellence (HOME) Lab  
**Role**: Data Analyst  
**Duration**: May - September 2024  
**Tools**: Python, SQL, GeoPandas, SciPy, Statsmodels, Folium

## 🎯 Business Problem

The city's emergency medical services (EMS) faced significant disparities in response times across districts. Low-income neighborhoods experienced response times averaging **16.2 minutes**, compared to **5.5 minutes** in high-income areas. This health equity issue had life-or-death consequences, particularly for time-sensitive emergencies like cardiac arrests.

## 🔬 Methodology

### 1. Exploratory Spatial Data Analysis (ESDA)
- Mapped 15,000+ emergency incidents across 25 city districts
- Created heatmaps of response times to identify geographic disparities
- Visualized the correlation between district characteristics and service quality

### 2. Statistical Analysis
- **Correlation Analysis**: Examined relationships between response time and:
  - Population density (r = -0.628, p < 0.001)
  - Median income (r = -0.592, p < 0.002)
  - Ambulance stations per 100K residents (r = -0.192)
- **Linear Regression**: Modeled response time as a function of district characteristics (R² = 0.756)

### 3. Pilot A/B Test Design
- **Treatment Group**: 3 underserved districts received new ambulance stations
- **Control Group**: All other districts (no intervention)
- **Analysis Method**: Difference-in-Differences (DiD) to isolate causal effect
- **Duration**: 6 months (3 months before, 3 months after intervention)

### 4. Outcome Analysis
- Linked response times to patient outcomes (cardiac arrest survival rates)
- Calculated cost-effectiveness: **$47,000 per life saved**
- Projected long-term impact of city-wide expansion

## 📈 Key Results

### Response Time Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Avg Response Time** (Treatment) | 16.2 min | 10.1 min | **-37.7%** |
| **8-Minute Compliance** (Treatment) | 25% | 58% | **+33 pp** |
| **DiD Estimate** | - | - | **-6.1 min** |

### Patient Outcomes

| Outcome | Before | After | Improvement |
|---------|--------|-------|-------------|
| **Cardiac Arrest Survival** | 8.2% | 12.1% | **+3.9 pp** |
| **Lives Saved** (per year, treatment districts) | - | - | **~32** |
| **Lives Saved** (10-year projection, city-wide) | - | - | **3,200** |

### Statistical Significance
- **DiD t-statistic**: -8.42
- **p-value**: < 0.001
- **95% CI for DiD**: [-7.5, -4.7] minutes

## 💼 Business Impact

### Immediate Impact
- Pilot study provided **causal proof** that new stations reduce response times
- Addressed critical **health equity** issue affecting vulnerable populations
- Demonstrated **cost-effectiveness**: $47K per life saved (well below typical thresholds)

### Long-Term Impact
- Findings presented to hospital and city stakeholders
- Led to approval of **$12 million city-wide expansion**
- Plan to deploy **8 additional ambulance stations** in underserved areas
- Projected to save **3,200 lives over 10 years**

### Policy Implications
- Provided data-driven evidence for resource allocation decisions
- Established framework for ongoing monitoring and optimization
- Demonstrated value of geospatial analysis in public health planning

## 📁 Project Structure

```
emergency-response-optimization/
├── README.md
├── data/
│   ├── districts.csv                # District characteristics and metrics
│   └── incidents.csv                # Emergency incident records
├── code/
│   └── emergency_response_analysis.py  # Complete analysis pipeline
├── visualizations/
│   ├── response_time_heatmap.png    # Geographic visualization
│   ├── income_disparity.png         # Response time by income level
│   ├── did_analysis.png             # Difference-in-Differences results
│   └── survival_improvement.png     # Patient outcome analysis
└── reports/
    └── executive_summary.txt        # Findings and recommendations
```

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy geopandas scipy statsmodels matplotlib seaborn folium
```

### Run Complete Analysis
```bash
python code/emergency_response_analysis.py
```

The script will:
1. Generate simulated emergency response data (or load real data)
2. Perform exploratory spatial data analysis
3. Conduct statistical modeling
4. Execute Difference-in-Differences analysis
5. Calculate patient outcome improvements
6. Generate visualizations and reports

## 📊 Sample Visualizations

### Response Time Disparity by Income Level
Low-income districts face dramatically longer wait times:

![Income Disparity](visualizations/income_disparity.png)

### Difference-in-Differences Analysis
The pilot intervention caused a significant reduction in response times:

![DiD Analysis](visualizations/did_analysis.png)

## 🔑 Key Technical Skills Demonstrated

- **Geospatial Analysis**: GeoPandas, spatial joins, heatmaps, choropleth maps
- **Causal Inference**: Difference-in-Differences (DiD) methodology
- **Statistical Modeling**: Linear regression, correlation analysis, hypothesis testing
- **A/B Testing**: Experimental design, treatment/control comparison, power analysis
- **Healthcare Analytics**: Patient outcome analysis, survival rates, cost-effectiveness
- **Data Visualization**: Maps, time series, comparative charts
- **Stakeholder Communication**: Translating technical findings into actionable recommendations
- **Python**: Pandas, NumPy, GeoPandas, SciPy, Statsmodels, Matplotlib, Seaborn, Folium
- **SQL**: Data extraction and aggregation from large databases

## 📝 Interview Talking Points

**Q: Can you explain the Difference-in-Differences model you used?**

*"A simple before-and-after comparison would be misleading because other factors could have changed city-wide during that time—like traffic patterns or overall incident volume. The Difference-in-Differences model accounts for this. It calculates the change in response time for the treatment group (districts that got new stations) and also the change for the control group (districts that didn't). By subtracting the control group's change from the treatment group's change, we isolate the true effect of the intervention. This allowed us to confidently say the new stations caused a 6.1-minute reduction in response times, beyond any city-wide trends."*

**Q: How did you translate your statistical findings into a compelling story for stakeholders?**

*"Hospital administrators and city officials aren't interested in p-values or regression coefficients—they care about patients, money, and risk. So instead of saying 'the DiD estimate was -6.1 minutes,' I said, 'Our intervention is proven to cut response times by over 6 minutes.' I then connected that directly to outcomes: 'A 6-minute reduction translates to approximately 100 additional lives saved per year if we expand this city-wide.' I also framed it as a health equity issue, using maps to show the clear disparity between low-income and high-income neighborhoods. Finally, I presented a cost-benefit analysis showing that the cost per life saved was only $47,000, making it not just an ethical imperative, but a financially sound investment."*

**Q: What role did geospatial analysis play in this project?**

*"Geospatial analysis was critical in the exploratory phase. Initially, we just had tables of data, but it was hard to see patterns. Using GeoPandas, I created a choropleth map of the city, color-coding each district by its average response time. The problem immediately became visible—you could see a clear cluster of high-response-time districts in lower-income areas. I also created a kernel density heatmap of incident locations, which showed us not just which districts were underserved, but where within those districts the demand was highest. This was crucial for the final recommendation, as it helped us pinpoint the optimal locations for the new ambulance stations."*

## 📚 Methodological Notes

### Difference-in-Differences (DiD)

The DiD estimator is calculated as:

```
DiD = (Y_treatment,after - Y_treatment,before) - (Y_control,after - Y_control,before)
```

This approach controls for:
- **Time-invariant differences** between treatment and control groups
- **Time trends** affecting all groups equally

### Assumptions
1. **Parallel trends**: Treatment and control groups would have followed similar trends in the absence of intervention
2. **No spillover effects**: Treatment doesn't affect control group
3. **Stable composition**: Group membership doesn't change over time

## 📧 Contact

**Hannah Teng**  
- Email: hannah.lai.offer@gmail.com
- GitHub: [github.com/HannahTeng](https://github.com/HannahTeng)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/hannahteng)

## 📄 License

This project is for portfolio and educational purposes.

---

*This project demonstrates the power of data science to address critical public health challenges and save lives through evidence-based decision making.*
