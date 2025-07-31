library(readxl)
AD=read.csv("AD.csv")
NAD=read.csv("NAD.csv")
AD1=AD[[6]]
NAD1=NAD[[6]]
n=1000
t_stat=numeric(1000)
#t-test for mean difference
for (i in 1:n){
 sampled_AD= sample(AD1,53,replace=TRUE)
   sampled_NAD= sample(NAD1,53,replace= TRUE)
   test=t.test(sampled_AD,sampled_NAD,var.equal = TRUE)
   t_stat[i]=test$statistic
}
mean(t_stat)
low=quantile(t_stat,0.025)
high=quantile(t_stat,0.975)
#confidence interval for t-test to check mean difference
low 
high

#F-test for variance of daily returns
set.seed((111))
Announcement_Day=AD1
NAD2 = sample(NAD1, size = 53)
Non_Announcement_Day=NAD2
var.test(Announcement_Day,Non_Announcement_Day)


