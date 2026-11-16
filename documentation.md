### **1.AI Solution**



SmartWatt is an electricity consumption and load forecasting system in Python for the energy sector (one of the sectors identified in the brief). It forecasts household electricity demand hour by hour and then translates that into practical advice (what time of day is peak, what time of day is cheapest to run your appliances, what your estimated bill will be) via a chatbot. It enables the 4IR in a developing-country context by putting data and AI in the hands of individual households and the municipality, rather than just large corporations. The utility and the municipality get demand visibility for planning and for avoiding overloads and load shedding. The household gets lower, more predictable bills. That's exactly what the brief was asking for in terms of cost effective growth through AI.



### **2.Business Objectives**



#### **Business Objectives**

* The best model achieves a at least 25% reduction in forecast error (in terms of both MAE and RMSE) over the baseline (i.e. the same hour yesterday).
* The forecast gives us the expected peak hour and the lowest demand hour over the next 24 hours.
* The chatbot answers the usual energy questions correctly.
* Target for a future pilot with real households: a 10% or more reduction in peak time usage through shifting (this figure is not measured in the prototype)



#### **Business success criteria**

* The best model reduces forecast error (in both MAE and RMSE) by at least 25% over the baseline (same hour yesterday).
* The model gets the peak times absolutely right.
* The chatbot gets the basic energy questions right.
* Estimated reduction in peak time use of 10% or more via the provision of alternative advice.



#### **Business background**

Electricity utilities and municipalities have to match supply to demand at all times. Bad forecasting leads to overloads, expensive emergency purchases and outages. Households have no means of understanding or controlling their own demand.



#### **Requirements**

* Python 3, historical consumption data, forecasting models, evaluation measures and a chatbot interface.
* Libraries: pandas, NumPy, scikit-learn, statsmodels, TensorFlow with Keras, joblib and matplotlib.
* One household dataset (2006 to 2010 from a house near Paris, France) might not be representative of South African homes.



#### **Constraints**



* One household dataset (2006-2010) wouldn't necessarily be representative of all households.
* Limited computing power for training deep learning models.
* There's no live smart meter feed in the prototype.
* Weather is not included unless it is added as an extra feature



#### **Risks and mitigation**



|**Risks**|**Mitigation**|
|-|-|
|Missing values in the data|Interpolation and cleaning|
|Overfitting|Chronological split, dropout, early stopping|
|Model does not generalise to other homes|Retrain on local smart meter data|
|User privacy|Anonymise and secure the data|
|Users ignore advice|Simple chatbot with clear savings figures|
|||





#### **Tools and Techniques**



|**Need**|**Tool and techniques**|
|-|-|
|Data handling|Python, pandas, NumPy|
|Time series analysis|statsmodels (seasonal decomposition)|
|Machine learning|scikit-learn (Random Forest)|
|Deep learning|TensorFlow with Keras (Long Short-Term Memory network)|
|Evaluation|Mean Absolute Error, Root Mean Squared Error, Mean Absolute Percentage Error, plots|



### **3. Problem Definition**



#### **What is the problem?**

Households in the developing countries cannot see or predict how much electricity they will use, or when. As a result, they run heavy appliances such as geysers, stoves and washing machines etc) during peak periods when the tariffs are highest and the grid is under the most stress.The bills are completely unpredictable and households get no advice on how to reduce them. The utilities and municipalities themselves have no forecasting at the household or neighbourhood level.The peaks in demand therefore simply overload the transformers and lines and that's what leads to outages and load shedding. It's all reactive rather than proactive.There is smart meter data available but it is not being used to generate forecasts or indeed simple advice that the average person can take on board.The fundamental problem is the absence of accurate, accessible electricity demand forecasting that would allow households to manage their consumption and would allow the energy providers to balance supply and demand.





#### **How relevant is it to the theme?**

The problem is directly relevant to the theme, an AI solution for industries. It sits in the energy industry, and artificial intelligence turns smart meter data into hourly demand forecasts and advice.



#### **How will solving the problem using artificial intelligence benefit the local municipality?**

Households are the main users. The benefit to a municipality is that you get the benefit of the model when it's applied across a large number of homes (e.g. using smart meter data at the neighbourhood level). The hourly forecast tells you when and where demand will peak so the municipality can plan supply, avoid transformer and line overloading and reduce emergency purchase costs. Shifting the household load away from peak times reduces the stress on the grid which in turn reduces the number of outages and the need for load shedding. The end result is that the residents save money which in turn improves service satisfaction and supports local economic development.

&#x09;

&#x09;

&#x09;

&#x09;



&#x09;

