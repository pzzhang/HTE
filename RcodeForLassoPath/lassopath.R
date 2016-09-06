
setwd("C:\\Users\\t-pezha\\OneDrive - Microsoft\\src\\second_order")

library(genlasso)


bb = read.csv("Bb.csv")
bb = bb[, -1]
bb = data.matrix(bb)
X = bb[, -ncol(bb)]
y = bb[, ncol(bb)]

D = read.csv("K.csv")
D = D[, -1]
D = data.matrix(D)

out = genlasso(y,X,D)