setwd("C:\\Users\\t-pezha\\OneDrive - Microsoft\\src\\bingad")

library(stringi)
library(data.table)
library(lubridate)

imps = fread("data\\Sample4096_111.txt", header=FALSE, sep="\t", colClasses = list(character = c(6,9)));
cols = c("ABTestingTypes", "ActualBidAmtUSD",               "AdUnitId",        "AdUnitTypeId",                "BusinessLocationIds",  "ChannelIds",    "CID",    "ClickCnt",           "DeviceModel", "DeviceOSId",                "DeviceOSName",           "DeviceOSVersion",        "DeviceTypeId", "DomainId",       "Dude",                "ECPIThresholdValue",  "FilterCB",           "FormCode",     "ImpressionCnt",             "MainLineReserve",                "MarketplaceClassificationId",   "MBDate",          "MBTime",          "MediumId",     "ML_Clicks",                "ML_Impressions",         "ML_Revenue", "ProbabilityOfClick",       "PropertyId",     "PublisherAccountId",                "PublisherId",    "Query",              "RankScoreUSD",             "RawQuery",     "RawBiddedSearchPages",                "RawSearchPages",        "RelatedToAccountId",  "RelationshipId",              "Revenue",         "RGUID",                "SearchEventId",             "SearchIG",        "SearchImpressionGuid",             "UserAgentId", "UserAgentString",                "AdLanguage",             "BucketId",       "CategoryId",    "SmartPricingQueryCategoryId",               "SubCategoryId",                "UserLocationCountryId",            "WebsiteCountry",         "AskedAdCnt",  "ReturnedAdCnt",                "SmartPricing", "SmartPricingDiscountPercentage",         "FormCodeClassification",           "LocationIds",                "QueryLocationIds",       "TrafficPartitionCode",  "FlightLine")
setnames(imps, cols)
#saveRDS(imps,'sample4096.rds')

#imps = readRDS('sample4096.rds')
impsNoCB = imps[FilterCB==0]  # filter out the Cache-buster traffic


imps_core = impsNoCB[, .(MBTime, Revenue, RawSearchPages,ABTestingTypes, DeviceModel, DeviceOSName, DeviceOSVersion, DeviceTypeId, FormCode, CategoryId, SmartPricingQueryCategoryId, SubCategoryId, LocationIds, QueryLocationIds)]

imps_core[,MBTime := parse_date_time2(MBTime, "mdYHMS")] # fast date time parsing convert the string to date time type

SeparateTime = parse_date_time2("4/15/2016 00:00:00", "mdYHMS") # the time to split into pre and post period
EndTime = parse_date_time2("5/6/2016 00:00:00", "mdYHMS") # the time to split into pre and post period

imps_core = imps_core[MBTime<EndTime]

#add pre post period indicator
imps_core[, Period:= ifelse(MBTime<SeparateTime, 'before', 'after')]
imps_core[, weekDay:= lubridate::wday(MBTime)]
imps_core[, Date:= lubridate::date(MBTime)]

setkey(imps_core, weekDay,Period)

#
#weekdayRpmlong = imps_core[, list(RPM = sum(Revenue)/sum(RawSearchPages)*1000), by = .(weekDay,Period)]
weekdayRpmlong = imps_core[, list(RPM = mean(Revenue)), by = .(weekDay,Period)]
weekdayRpmWide = dcast(weekdayRpmlong, weekDay~Period, value.var = 'RPM')
weekdayRpmWide[, diff:=after/before-1]
weekdayRpmWide

# 
summary(imps_core)
#count levels
lapply(imps_core, function(x) length(unique(x)))

imps_core[, list(N=.N), keyby = .(Period)]
imps_core[, list( rpm= sum(Revenue)/sum(RawSearchPages)*1000, se = sd(Revenue)/sqrt(.N), NRow = .N, SearchPgs= sum(RawSearchPages) ),keyby=.(DeviceOSName, Period)]


# remove andoid: 
imps_dv = copy(imps_core)
imps_dv[, IsAndroid:= DeviceOSName=='Android']
imps_dv[, list( rpm= sum(Revenue)/sum(RawSearchPages)*1000,NRow = .N, SearchPgs= sum(RawSearchPages) ),keyby=.(IsAndroid, Period)]
