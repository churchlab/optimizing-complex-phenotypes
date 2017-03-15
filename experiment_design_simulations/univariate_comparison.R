library(data.table)
library(scales)

### load doubling times ###
dbl.fn <- '~/Dropbox/Projects/fix-recoli-git/experiment_design_simulations/results/supp_fig4a_data/wgs_samples_doubling_times.csv'
dbl.dt <- data.table(read.csv(dbl.fn, header=F, col.names=c('doubling')))

### load samples ###
smp.fn <- '~/Dropbox/Projects/fix-recoli-git/experiment_design_simulations/results/supp_fig4a_data/wgs_samples.csv'
smp.dt <- data.table(read.csv(smp.fn, header=F))

# add samples
smp.dt[, sample := 1:.N]

# add doubling times
smp.dt[, dbl := dbl.dt]

# melt on samples
smp.dt <- melt(smp.dt, id.var=c('dbl','sample'), variable.name='snv')

# convert snvs to numbers
smp.dt[, snv := as.numeric(substr(as.character(snv), 2, nchar(as.character(snv))))]

### load effects ###
eff.fn <- '~/Dropbox/Projects/fix-recoli-git/experiment_design_simulations/results/supp_fig4a_data/snp_effects.csv'
eff.dt <-  data.table('eff'=read.csv(eff.fn, header=F)[,1])

# add snp number
eff.dt[, snv := 1:.N]

### perform gwas

snv_indiv_lm <- smp.dt[,
  structure(
    as.list(summary(lm(dbl ~ value))$coefficients[c(2,8)]), 
    .Names=c('coeff','pval')), 
  by=snv]

ggplot(snv_indiv_lm[eff.dt, on='snv'], aes(
    x=coeff+1, 
    y=eff, 
    fill=-log10(pval/nrow(eff.dt)))) + 
  geom_point(shape = 21, size=4, color='black') +
  scale_fill_distiller(palette='RdYlBu', name='Bonferroni\nCorrected\n-log10(p)') +
  labs(x='Linear Model Coefficient (GWAS)', y='True SNP Effect') +
  geom_abline(intercept=0, slope=1, linetype=2) +
  theme_minimal()

### perform gwas on real data

source('rstudio/fxrc_common.R')

# Load Data  
if (file.exists('rstudio/fxrc.rdata')) {
  load('rstudio/fxrc.rdata')
} else {
  source('rstudio/fxrc_loaddata.R')
}
source('rstudio/fxrc_common.R')

#use MJL measurements
use.measure = 'mjl'
start_dbl_time <- dbltime.minmax.dt[name==use.measure, start]
wt_dbl_time <- dbltime.minmax.dt[name==use.measure, wt]
exp1.dt <- exp1.dt[fetch_dbl_times(use.measure), on='well']

# scale MJL measurements to construction measurements to match figure 3
# (measuring best clone here against finals, this scale operation is warranted)
exp1.dt[,
  doubling_scaled := rescale(
    doubling_time,
    to=c(c321.constr.dbltime,
      construction.dbltime.dt[class=='DPRFA',doubling]))]

snv_indiv_lm <- exp1.dt[sum_signals > 0,
  structure(
    as.list(
      summary(lm(doubling_time ~ signal_relative_to_C321))$coefficients[c(2,8)]), 
    .Names=c('coeff','pval')), by=c('snv_lbl')][!is.na(coeff)][order(pval)]

# bonferroni
snv_indiv_lm[, pval := pmin(1, pval*.N)]

# merge with model coeffs
snv_indiv_lm <- model_coeff_data[
  snv_indiv_lm, ,on='snv_lbl'][order(pval)][pval < 1 | fraction_improvement > 0]

write.csv(snv_indiv_lm,
  '~/Dropbox/Projects/fix-recoli-git/rstudio/gwas_pval_table.csv')
