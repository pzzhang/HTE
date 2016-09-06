setwd("C:\\Users\\t-pezha\\OneDrive - Microsoft\\src\\bingads_multiplemetrics")

library(stringi)
library(data.table)
library(lubridate)
require(utils)

imps = fread("data\\metric-convert2.csv")
imps$Date = as.Date(imps$Date, "%m/%d/%Y")
setkey(imps, Metric, Date)

# construct a grid and full the table
values = lapply(imps[, .(Metric, Date)], function(x) unique(x))
datagrid = data.table(expand.grid(values))
setkey(datagrid, Metric, Date)
imps = merge(datagrid, imps, all.x=TRUE, by=c("Metric", "Date"))
setkey(imps, Metric, Date)

write.csv(imps, file = "data\\bingads_mm.csv")