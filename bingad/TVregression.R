setwd("C:\\Users\\t-pezha\\OneDrive - Microsoft\\src\\bingad")

library(stringi)
library(data.table)
library(lubridate)
require(utils)

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
imps_core[, DeviceOSName:= ifelse(nchar(DeviceOSName), DeviceOSName, 'unknown')]

setkey(imps_core, Period, RawSearchPages)

#construct two TB: one for before, one for after
TBbef = imps_core['before',]
setkey(TBbef, DeviceOSName)
TBaft = imps_core['after',]
setkey(TBaft, DeviceOSName)
TBbef = TBbef[, list( rpm = sum(Revenue)/sum(RawSearchPages)*1000, rev_var = ifelse(.N>1, 1e6*var(Revenue), 0), var_rev = 1.0/.N, NRow = .N, SearchPgs= sum(RawSearchPages) ), by=.(DeviceOSName, DeviceTypeId, CategoryId, weekDay)]
TBaft = TBaft[, list( rpm = sum(Revenue)/sum(RawSearchPages)*1000, rev_var = ifelse(.N>1, 1e6*var(Revenue), 0), var_rev = 1.0/.N, NRow = .N, SearchPgs= sum(RawSearchPages) ), by=.(DeviceOSName, DeviceTypeId, CategoryId, weekDay)]
# simple variance for revenue per search
varbefore = TBbef[, sum(rev_var*(NRow-1))]/TBbef[, sum(NRow-1)]
varafter = TBaft[, sum(rev_var*(NRow-1))]/TBaft[, sum(NRow-1)]
vartotal = (TBbef[, sum(rev_var*(NRow-1))] + TBaft[, sum(rev_var*(NRow-1))])/(TBbef[, sum(NRow-1)] + TBaft[, sum(NRow-1)])


# calucalate variance
TotalCountsBef = TBbef[, sum(NRow)]
TotalPageViewBef = TBbef[, sum(SearchPgs)]
TotalCountsAft = TBaft[, sum(NRow)]
TotalPageViewAft = TBaft[, sum(SearchPgs)]
TBbef[, c("weights", "rho", "pvweights", "logvar_rev") := .(NRow/TotalCountsBef, (SearchPgs+1)/(NRow+2), SearchPgs/TotalPageViewBef, var_rev/rpm^2)]
TBaft[, c("weights", "rho", "pvweights", "logvar_rev") := .(NRow/TotalCountsAft, (SearchPgs+1)/(NRow+2), SearchPgs/TotalPageViewAft, var_rev/rpm^2)]
rhowtotalBef = TBbef[, sum(rho*weights)]
rhowtotalAft = TBaft[, sum(rho*weights)]

#calculate the variance
varlogtotalBef = TBbef[, sum(weights*weights*rho*(1-rho)/NRow)]/(rhowtotalBef)^2
varlogtotalAft = TBaft[, sum(weights*weights*rho*(1-rho)/NRow)]/(rhowtotalAft)^2
TBbef[, varlogpv:= (1-rho)/(SearchPgs+1e-8) - 2*weights*(1-rho)/(NRow*rhowtotalBef) + varlogtotalBef]
TBaft[, varlogpv:= (1-rho)/(SearchPgs+1e-8) - 2*weights*(1-rho)/(NRow*rhowtotalAft) + varlogtotalAft]
TBbef[, varpv:= (weights*rho/rhowtotalBef)^2*varlogpv]
TBaft[, varpv:= (weights*rho/rhowtotalAft)^2*varlogpv]

#filter out the small segments, calculate variance for "before"
TBbef = TBbef[SearchPgs>5]
TBaft = TBaft[SearchPgs>5]
TBbefshort = TBbef[, .(DeviceOSName, DeviceTypeId, CategoryId, weekDay, pvweights, varpv, varlogpv, rpm, var_rev, logvar_rev)]
TBaftshort = TBaft[, .(DeviceOSName, DeviceTypeId, CategoryId, weekDay, pvweights, varpv, varlogpv, rpm, var_rev, logvar_rev)]

#calculate the delta and its variance
TBdelta = merge(TBbefshort, TBaftshort, by = c("DeviceOSName", "DeviceTypeId", "CategoryId", "weekDay"), suffixes=c(".B", ".A"))
TBdelta[, c("delta", "logdelta", "mdelta", "mlogdelta"):= .((pvweights.A-pvweights.B)*1e3, log(pvweights.A/pvweights.B), 1e-6/(varpv.A+varpv.B), 1/(varlogpv.A+varlogpv.B))]
TBdelta[, c("rdelta", "rlogdelta", "rmdelta", "rmlogdelta"):= .(rpm.A-rpm.B, log(rpm.A/rpm.B), 1/(var_rev.A+var_rev.B), 1/(logvar_rev.A+logvar_rev.B))]
TB_tvreg = TBdelta[, .(DeviceOSName, DeviceTypeId, CategoryId, weekDay, delta, logdelta, mdelta, mlogdelta, rdelta, rlogdelta, rmdelta, rmlogdelta)]
setkey(TB_tvreg, DeviceOSName, DeviceTypeId, CategoryId, weekDay)

#make the data a grid
values = lapply(TB_tvreg[, .(DeviceOSName, DeviceTypeId, CategoryId, weekDay)], function(x) unique(x))
datagrid = data.table(expand.grid(values))
setkey(datagrid, DeviceOSName, DeviceTypeId, CategoryId, weekDay)
TB_tvreg = merge(datagrid, TB_tvreg, all.x=TRUE, by=c("DeviceOSName", "DeviceTypeId", "CategoryId", "weekDay"))
setkey(TB_tvreg, DeviceOSName, DeviceTypeId, CategoryId, weekDay)

write.csv(TB_tvreg, file = "data/volume.csv")
