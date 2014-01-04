library("arules")
datingdata <- read.csv(file.choose())
head(datingdata)
#Solution to Answer.3a 

relationshipdata<-subset(datingdata,datingdata$Outcome==1)
nrow(relationshipdata)/nrow(datingdata)*100

#Solution to Answer.3b
diff<-datingdata$Male.Age - datingdata$Female.Age
#x is the years of difference where I consider same age
x<-2
datingdata$smartage<-ifelse(diff > abs(x) ,"Female Younger",
                 ifelse(abs(diff) == abs(x) |abs(diff) == 0 , "Same Age","Female Older" ))
table(datingdata$smartage)
#Younger Female Proportion
length(relationshipdata$Outcome[relationshipdata$smartage=='Female Younger'])/nrow((relationshipdata))

#Older Female Proportion
length(relationshipdata$Outcome[relationshipdata$smartage=='Female Older'])/nrow((relationshipdata))


#same age
length(relationshipdata$Outcome[relationshipdata$smartage=='Same Age'])/nrow((relationshipdata))

# Solution to answer 3c
range<-seq(0,5)
diff<-datingdata$Male.Age - datingdata$Female.Age
for(x in range){

smartage<-ifelse(diff > abs(x) ,"Male Older",
                 ifelse(abs(diff) == abs(x) |abs(diff) == 0 , "Same Age","Male Younger" ))
finaldata<-data.frame(datingdata$Outcome,as.factor(smartage))
finaldata$datingdata.Outcome<-as.factor(finaldata$datingdata.Outcome)
data2 <- as(finaldata, "transactions")
dating.rules <- apriori(data2,
                        parameter = list(support=0.01, confidence=0.01,
                                         minlen = 2, maxlen = 2))

rules.sub <- subset(dating.rules,subset=rhs %in% "datingdata.Outcome=1")
print(x)
print(inspect(sort(rules.sub,by ="confidence") ))
}
