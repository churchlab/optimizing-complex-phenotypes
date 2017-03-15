library(data.table)
library(ggplot2)
library(stringr)
library(gridExtra)

### Load raw data

data.fn <- '~/Dropbox/Projects/fix-recoli-git/experiment_design_simulations/results/sim_results_num_strains_vs_num_snps_identified.csv'
data.dt <- data.table(read.csv(data.fn))

### Impute data functions
# Impute a z single variable over a set of 3 independent variables.
loess_impute_val_3d <- function (dt, x1, x2, x3, z, inc=10) {
  
  #https://www.r-statistics.com/2016/07/
  #   using-2d-contour-plots-within-ggplot2-to-
  #   visualize-relationships-between-three-variables/
  message('Imputing ', x1,' . ',x2, ' . ',x3,' on ',z)
  f <- as.formula(paste(z, '~', x1,'+',x2,'+',x3))
  data.loess <- loess(f, data = dt)
  
  # Create a sequence of incrementally increasing (by inc) values for x,y,c
  if (length(inc) == 1) {
    inc <- rep(inc, 3)
  }
  
  x1grid <-  seq(min(dt[, x1, with=F]), max(dt[, x1, with=F]), inc[1])
  x2grid <-  seq(min(dt[, x2, with=F]), max(dt[, x2, with=F]), inc[2])
  x3grid <-  seq(min(dt[, x3, with=F]), max(dt[, x3, with=F]), inc[3])

  # Generate a dataframe with every possible combination of x and y
  data.fit <-  data.table(expand.grid(x1 = x1grid, x2 = x2grid, x3 = x3grid))
  names(data.fit) <- c(x1, x2, x3)
  # Feed the dataframe into the loess model and receive a matrix output with estimates of 
  # acceleration for each combination of x and y
  # Transform data to long form
  mtrx.melt <-  data.table(
    data.fit, 
    z=predict(data.loess, newdata = data.fit),
    key=c(x1,x2,x3))
  names(mtrx.melt) <- c(x1, x2, x3, z)
  
  # set imputation max/min (this will screw up any values not on 0-1 scale...)
  mtrx.melt[get(z) < 0, c(z) := list(0)]
  mtrx.melt[get(z) > 1, c(z) := list(1)]
  
  return(mtrx.melt)
}

# combine seperately imputed z vars into a single dt
loess_impute_val_all <- function(dt, x1, x2, x3, inc=1) {
  z_set <- setdiff(names(dt), c(x1, x2, x3))
  z_impute <- function(z) loess_impute_val_3d(dt, x1, x2, x3, z, inc)
  return(Reduce(merge,lapply(z_set,z_impute)))
}

# plot a single set of plots with one axis (const) held constant over the
# whole imputation space of the other two axes
plot_loess_impute_set <- function(
    data.dt,
    mtrx.melt,
    x,
    y,
    const,
    z,
    const.val,
    cut.range=F,
    cut.num=11,
    col.dir=1,
    show.pts=T) {

  x_range <- c(min(mtrx.melt[,x, with=F]), max(mtrx.melt[,x, with=F]))
  y_range <- c(min(mtrx.melt[,y, with=F]), max(mtrx.melt[,y, with=F]))
  
  z_cols <- c(paste0('gwas_',z), paste0('lm_',z))
  
  if (paste0('enrichment_',z) %in% names(data.dt)) {
    z_cols <- c(z_cols, paste0('enrichment_',z))
  }

  melt.data.dt <- melt(data.dt[, c(x,y,z_cols), with=F], id.vars=c(x,y))
  
  mtrx.melt.copy <- melt(mtrx.melt[, 
    c(x, y, z_cols), with=F], id.vars=c(x,y))
  
  # remove NA values
  mtrx.melt.copy <- mtrx.melt.copy[!is.na(value),]
  melt.data.dt <- melt.data.dt[!is.na(value),]
  

  if (cut.range == F) {
    cut.range <- c(
      min(c(melt.data.dt$value, mtrx.melt.copy$value, 1))-0.01, 
      max(c(melt.data.dt$value, mtrx.melt.copy$value, 1))+0.01)
  }

  mtrx.melt.copy[, 
    cut_z := cut(value,
      seq(cut.range[1],cut.range[2],length.out=cut.num),
      right=F,
      include.lowest=T)]
  
  melt.data.dt[,
    cut_z := cut(value,
      seq(cut.range[1],cut.range[2],length.out=cut.num),
      right=F,
      include.lowest=T)]

  ggplot() +
    geom_raster(
      data=mtrx.melt.copy,
      aes_string(x = x, y = y, fill= 'cut_z'),
      alpha=0.7) +
    geom_point(
      data=melt.data.dt, 
      aes_string(x = x, y = y, color = 'cut_z'), 
      position=position_jitter(diff(x_range)/50, diff(y_range)/50)) +
    scale_color_brewer(
      palette='RdYlBu', name=z, direction=col.dir, drop=F) +
    scale_fill_brewer(palette='RdYlBu', name=z, direction=col.dir, drop=F) +
    facet_grid(~variable) +
    ggtitle(paste(z, 'WITH', const, '=', const.val, 'ON', x, 'VS', y)) +
    theme_bw()
}

plot_col_set <- function(
    data.dt, indep.vars, z, format='pdf', inc=c(5,4,1)) {
  
  x1 <- indep.vars[1]
  x2 <- indep.vars[2]
  x3 <- indep.vars[3]
  
  message(sprintf("x1='%s'; x2='%s'; x3='%s'; z='%s'", x1, x2, x3, z))

  # grab all columns in this col set
  z_cols <- c(paste0('gwas_',z), paste0('lm_',z))
  if (paste0('enrichment_',z) %in% names(data.dt)) {
    # add enrichment if available
    z_cols <- c(z_cols, paste0('enrichment_',z))
  }
  
  # subset the original data frame on the relevant cols
  data.subset.dt <- data.dt[, c(x1,x2,x3,z_cols), with=F]
  
  mtrx.melt <- loess_impute_val_all(
    data.subset.dt,
    x1, x2, x3, inc)
  
  plot.fn <- function(x, y, const, const.val, z) paste0(
    output.dir, '/',
    x, '.', y, '.',z,'.',
    const, '-',sprintf('%04d',const.val),'.',format)
  
  save_single_plot <- function(x, y, const, const.val) ggsave(
    filename=plot.fn(x, y, const, const.val, z),
    plot=plot_loess_impute_set(
      data.subset.dt[get(const)==const.val, c(x,y,const,z_cols), with=F],
      mtrx.melt[get(const)==const.val, c(x,y,const,z_cols), with=F],
      x,
      y,
      const,
      z,
      const.val,
      col.dir=1),
    width=(6*length(z_cols)+3),
    height=8,
    units='in',
    dpi=100)
  
  # function for making all plots with 1 axis constant
  plot_set_with_const <- function (x,y, const) lapply(
    as.numeric(unlist(unique(data.subset.dt[, const, with=F]))),
    function(const.val) {
      message(sprintf("    x='%s'; y='%s'; const='%s'; const.val=%s",
        x, y, const, const.val))
      save_single_plot(x, y, const, const.val)
    }
  )
  
  # make all 3 plot sets
  plot_set_with_const(x1, x2, x3)
  plot_set_with_const(x3, x2, x1)
  #plot_set_with_const(x1, x3, x2)
  
  # function for making gifs
  make_gifs <- function(x,y,const) {
    fn <- paste0(output.dir,'/',x,'.',y,'.',z,'.',const)
    imagemagick_cmd <- paste0(
      'convert -delay 100x100 -loop 0 ',fn,'*.',format,' ',
      fn,'.gif')
    system(imagemagick_cmd)
    system(paste0('convert -delay 100x100 ',fn,'.gif ',fn,'.gif'))
  }
  
  # make gifs for 3 plot sets
  make_gifs(x1, x2, x3)
  make_gifs(x3, x2, x1)
  #make_gifs(x1, x3, x2)
}

#### plot and save all set variable combos
indep.vars <- c('num_snps_considered', 'num_samples', 'num_snps_with_effect')

output.dir <- '~/Dropbox/Projects/fix-recoli-git/experiment_design_simulations/r_plotting/plots'

col.sets <- setdiff(names(data.dt), indep.vars)
col.sets <- col.sets[grep('(gwas|lm)',col.sets)]
col.sets <- unique(sub('(gwas|lm)_','',col.sets))

# we only want to plot a subset of the columns:
#col.sets <- col.sets[c(1,7,8,9,10,14,15,16)]
col.sets <- col.sets[c(7,8,15,16)]

plots_for_col_set <- function(z) {

  plot_col_set(data.dt, indep.vars, z, format='pdf')
}

# MAKE ALL THE PLOTS AND GIFS!
lapply(col.sets, plots_for_col_set)

