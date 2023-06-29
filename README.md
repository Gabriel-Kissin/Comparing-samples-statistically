# Comparing samples statistically

#### The aim of this notebook is to provide some basic graphical and numerical methods for statistical comparison of two samples.

Because a picture is worth a thousand words, here is one example of what the notebook creates, to compare data from Normal and Uniform distributions:

![Normal vs Uniform distributions](https://github.com/Gabriel-Kissin/Comparing-samples-statistically/assets/118690308/5bb16985-b2a6-4028-914f-5a98e237d5e7)


And numerical statistical tests for differences, for the same two samples:
```
                      test     statistic    pvalue  statistic_location     statistic_sign
Ttest_indResult  Ttest_ind -4.313415e-15  1.000000                 NaN                NaN  
BartlettResult    Bartlett -3.327339e-13  1.000000                 NaN                NaN  
KstestResult        Kstest  6.200000e-02  0.006253           -0.824785               -1.0 
```

See the notebook for one-sided tests and for detailed descriptions of all the features on the plots above.
