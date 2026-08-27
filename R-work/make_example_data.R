set.seed(7)
n <- 400
d <- data.frame(
  X            = 1:n,
  age          = round(rnorm(n, 38, 12)),
  gender       = sample(c("M","F"), n, TRUE),
  income       = round(rlnorm(n, 10.5, .5)),
  region       = sample(c("north","south","east","west"), n, TRUE),
  service      = sample(1:5, n, TRUE),
  wait_minutes = round(rexp(n, 1/12), 1),
  price_fair   = sample(1:5, n, TRUE),
  visits       = rpois(n, 4)
)
lp <- -2 + .9*d$service + .5*d$price_fair - .06*d$wait_minutes + .015*(d$age-38)
p  <- 1/(1+exp(-lp))
d$satisfied <- ifelse(runif(n) < p, "yes", "no")
d$income[sample(n, 15)] <- NA          # exercise imputation
d$region[sample(n, 8)]  <- NA
write.csv(d, "example_data.csv", row.names = FALSE)
cat("rows", nrow(d), "cols", ncol(d), "\n"); print(table(d$satisfied))
