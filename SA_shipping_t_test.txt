from scipy import stats
import pandas as pd

# Sample data for two groups
df = pd.read_excel('SA_shipping_data.xlsx')
i8000 = df[df['Ship Model'].str[12] == '8']
i9000 = df[df['Ship Model'].str[12] == '9']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(i8000['Alarm Signal'], i9000['Alarm Signal'])

# Output the results
print("T-statistic:", t_statistic)
print("P-value:", p_value)

# Interpret the results
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the two groups.")