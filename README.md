# Description
This dataset is developed from an open USGS dataset for water quality monitoring at the Port of Albany (New York) on the Hudson River ([USGS Station Website](https://waterdata.usgs.gov/monitoring-location/01359165/#parameterCode=00010&period=P7D&showMedian=true)). The data have been processed and minimally QC'd. 

The columns are:

-  `datetime_utc`: timestamp for observations given in UTC
-  `wtempc`: water temperature in degrees Celsius
-  `atempc`: air temperature at water surface in degrees Celsius
-  `winddir_dcfn`: wind direction in degrees clockwise from North
-  `precp_in`: precipitation in inches
-  `relh_pct`: relative humidity (percent of saturation)
-  `spc`: specific conductivity (microsiemens/centimeter)
-  `dox_mgl`: dissolved oxygen in milligrams per liter
-  `ph`: pH of water in standard units (SU, feasible range of 0-14)
-  `windgust_knots`: speed of wind gusts in knots
-  `wse1988`: water surface elevation in NAVD88 datum
-  `wvel_fps`: water velocity in feet per second
-  `mbars`: atmospheric pressure in millibars
-  `windspeed_knots`: average wind speed in knots
-  `par`: photosynthetically available radiation (millimoles of photons per square meter)
-  `turb_fnu`: water turbidity (formazin nephelometric units)

More details on each of these measurement units are obtainable at the station website cited above.

# Tasks
At least three, but no more than four, of the following tasks should be completed. Provide your work in one or more python notebooks, organized as you see best fit. Don't spend more than two hours in total on these tasks - the purpose is for you to show us how you get familiar with the dataset and begin to explore modeling the data, not to create a perfect model. You will have 15 minutes to present your responses to us in the interview, after which we will engage in a Q&A discussion.

> Note: The `.gitignore` file may need to be edited in order to commit your work. It is expected that you know how to do this and what needs to be changed by examining the current file.

## Task 1 - Document your development tools
Use a method of your choice to document the tools you've used in generating your responses, with an aim toward reproducibility of your work. Be prepared to explain your choice.

## Task 2 - Describe the development data set
Minimally, generate a table describing the empirical statistics of the development dataset in the file `dev.csv`. Use your tool of choice to do so. Produce another table summarizing the completeness of the record for each feature. Generate a graphic summarizing these descriptions that could be used as part of a logging procedure in a featurization pipeline. Assume that these procedures will be re-used as part of an overall machine-learning pipeline, and be prepared to comment on how this fact influences your choices in developing your response.

## Task 3 - Explore interesting characteristics of the development data set
Be creative. This is real-world environmental data. What is interesting about it to you? What features are surprising or exhibit characteristics that may cause issues when used in modeling? Select a subset of features that exhibit such a characteristic and explore it further (be prepared to discuss your findings).

## Task 4 - Causal analysis
This task is intended to assess your understanding of the environmental science domain. Which of the features is - in theory - causally influenced by others in the dataset? What might a causal graph for such a hypothesis look like? What tools would you consider using to test this? It's okay if you have not used them, but be prepared to explain why you are interested in applying them to this question. Discuss your methodology (resources queried, assessment process) for formulating a response to this question, and what you may have learned in the process.

> Note: If you do implement something while responding to this prompt, please save it in case we have time to discuss in more detail.  

## Task 5 - Model development
Develop a model of your choice (i.e., you select the learning task, the target feature, and the input features) and train this model using a library of your choice. Use best practices for training this model and report your training, validation and test scores. Discuss any conclusions you are confident in drawing from this exercise. Is your model ready for production use? Why or why not? 
