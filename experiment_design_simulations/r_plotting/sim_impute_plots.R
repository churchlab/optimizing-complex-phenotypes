library(data.table)
library(ggplot2)
library(stringr)
library(gridExtra)

### Load raw data

data.fn <- '~/Dropbox/Projects/fix-recoli-git/experiment_design_simulations/results/sim_results_num_strains_vs_num_snps_identified.csv'
data.dt <- data.table(read.csv(data.fn))

### Impute data functions

loess_impute_val <- function (dt, x, y, z, inc=10) {
  
  #https://www.r-statistics.com/2016/07/
  #   using-2d-contour-plots-within-ggplot2-to-
  #   visualize-relationships-between-three-variables/
  message(x,' ',y)
  message('columns: ',paste(names(dt),collapse=' '))
  print(unique(dt[, x, with=F]))
  print(unique(dt[, y, with=F]))
  f <- as.formula(paste(z, '~', x,'+',y))
  data.loess <- loess(f, data = dt)
  
  # Create a sequence of incrementally increasing (by 0.3 units) values for both x and y
  xgrid <-  seq(min(dt[, x, with=F]), max(dt[, x, with=F]), inc)
  ygrid <-  seq(min(dt[, y, with=F]), max(dt[, y, with=F]), inc)
  
  # Generate a dataframe with every possible combination of x and y
  data.fit <-  expand.grid(x = xgrid, y = ygrid)
  names(data.fit) <- c(x,y)
  # Feed the dataframe into the loess model and receive a matrix output with estimates of 
  # acceleration for each combination of x and y
  mtrx3d <-  predict(data.loess, newdata = data.fit)
  
  # Transform data to long form
  mtrx.melt <- melt(mtrx3d, id.vars = c(x, y), measure.vars = z)
  names(mtrx.melt) <- c(x, y, z)
  # Return data to numeric form
  mtrx.melt[,x] <- as.numeric(str_sub(mtrx.melt[,x], str_locate(mtrx.melt[,x], "=")[1,1] + 1))
  mtrx.melt[,y] <- as.numeric(str_sub(mtrx.melt[,y], str_locate(mtrx.melt[,y], "=")[1,1] + 1))
  mtrx.melt <- data.table(mtrx.melt, key=c(x,y))
  return(mtrx.melt)
}

loess_impute_val_3d <- function (dt, x, y, const, z, inc=10) {
  
  #https://www.r-statistics.com/2016/07/
  #   using-2d-contour-plots-within-ggplot2-to-
  #   visualize-relationships-between-three-variables/
  message('Imputing ', x,' . ',y, ' . ',const,' on ',z)
  f <- as.formula(paste(z, '~', x,'+',y,'+',const))
  data.loess <- loess(f, data = dt)
  
  # Create a sequence of incrementally increasing (by inc) values for x,y,c
  if (length(inc) == 1) {
    inc <- rep(inc, 3)
  }
  
  xgrid <-  seq(min(dt[, x, with=F]), max(dt[, x, with=F]), inc[1])
  ygrid <-  seq(min(dt[, y, with=F]), max(dt[, y, with=F]), inc[2])
  cgrid <-  seq(min(dt[, const, with=F]), max(dt[, const, with=F]), inc[3])

  # Generate a dataframe with every possible combination of x and y
  data.fit <-  expand.grid(x = xgrid, y = ygrid, c = cgrid)
  names(data.fit) <- c(x,y)
  # Feed the dataframe into the loess model and receive a matrix output with estimates of 
  # acceleration for each combination of x and y
  mtrx3d <-  predict(data.loess, newdata = data.fit)
  
  # Transform data to long form
  mtrx.melt <- melt(mtrx3d, id.vars = c(x, y), measure.vars = z)
  names(mtrx.melt) <- c(x, y, z)
  # Return data to numeric form
  mtrx.melt[,x] <- as.numeric(str_sub(mtrx.melt[,x], str_locate(mtrx.melt[,x], "=")[1,1] + 1))
  mtrx.melt[,y] <- as.numeric(str_sub(mtrx.melt[,y], str_locate(mtrx.melt[,y], "=")[1,1] + 1))
  mtrx.melt <- data.table(mtrx.melt, key=c(x,y))
  return(mtrx.melt)
}

loess_impute_val_all <- function(dt, x, y, inc=2.5) {
  z_set <- setdiff(names(dt), c(x,y))
  z_impute <- function(z) loess_impute_val(dt, x, y, z, inc)
  return(Reduce(merge,lapply(z_set,z_impute)))
}

loess_impute_gwas_lm_set <- function(
    mtrx.melt, 
    data.dt,
    z,
    const='',
    const.val='',
    x='num_snps_with_effect', 
    y='num_samples',
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
  # set NA values to 1 explicitly
  melt.data.dt[is.na(value), value := 1]
  
  if (cut.range == F) {
    cut.range <- c(0, max(c(melt.data.dt$value, mtrx.melt.copy$value, 1))+0.01)
  }
    
  mtrx.melt.copy <- melt(mtrx.melt[, 
    c(x, y, z_cols), with=F], id.vars=c(x,y))

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
    data.dt, col.set, x, y, const, const.val, format='pdf') {
  
  plot.fn <- paste0(
    output.dir, '/',
    x, '.', y, '.',col.set,'.',
    const, '-',sprintf('%04d',const.val),'.',format)
  message(paste('Printing',plot.fn))
  
  z_cols <- c(paste0('gwas_',col.set), paste0('lm_',col.set))
  
  if (paste0('enrichment_',col.set) %in% names(data.dt)) {
    z_cols <- c(z_cols, paste0('enrichment_',col.set))
  }
  
  message(sprintf('x=%s, y=%s, const=%s', x, y, const))
  data.subset.dt <- data.dt[get(const)==const.val, c(x, y, z_cols), with=F]

  print(unique(data.dt[, x, with=F]))
  print(unique(data.dt[, y, with=F]))
  
  print(unique(data.subset.dt[, x, with=F]))
  print(unique(data.subset.dt[, y, with=F]))
    
  mtrx.melt <- loess_impute_val_all(data.subset.dt, x, y, inc)
  
  
  ggsave(
    filename=plot.fn,
    plot=loess_impute_gwas_lm_set(
      mtrx.melt,
      data.dt[get(const)==const.val, c(x, y, z_cols), with=F],
      x=x,
      y=y,
      z=col.set,
      const=const,
      const.val=const.val,
      col.dir=1),
    units='in',
    dpi=100)
}

#### plot and save all set variable combos
#lapply(col.sets, plot_col_set)

indep.vars <- c('num_snps_considered', 'num_samples', 'num_snps_with_effect')

output.dir <- '~/Dropbox/Projects/fix-recoli-git/experiment_design_simulations/r_plotting/plots'

col.sets <- setdiff(names(data.dt), indep.vars)
col.sets <- col.sets[grep('(gwas|lm)',col.sets)]
col.sets <- unique(sub('(gwas|lm)_','',col.sets))

# we only want to plot a subset of the columns:
col.sets <- col.sets[c(1,7,8,9,10,14,15,16)]

plot_all_for_xy_pair <- function(data.dt, x, y) {

  const <- setdiff(indep.vars, c(x,y))
  const.vals <- as.numeric(unlist(unique(data.dt[, const, with=F])))

  message(sprintf('Plotting all for x=%s, y=%s, const=%s', x, y, const))

  outer(seq_along(const.vals), seq_along(col.sets),
    FUN= Vectorize(function(i,j) plot_col_set(
      data.dt, col.sets[j], x, y, const, const.vals[i], format='png')))

  make_gifs <- function(x,y,col.set) {
    fn <- paste0(output.dir,'/',x,'.',y,'.',col.set,'.',const)
    imagemagick_cmd <- paste0(
      'convert -delay 100x100 -loop 0 ',fn,'*.png',
      fn,'.gif')
    system(paste0('convert -delay 100x100 ',fn,'.gif ',fn,'.gif'))
  }
    
  lapply(col.sets, function (col.set) make_gifs(x,y,col.set))
}

### plot all xy pairs
plot_all_for_xy_pair(data.dt, indep.vars[1], indep.vars[2])
plot_all_for_xy_pair(data.dt, indep.vars[2], indep.vars[3])
plot_all_for_xy_pair(data.dt, indep.vars[1], indep.vars[3])

#### run single plots

# loess_impute_plot(
#   mtrx.melt, data.dt, z='enrichment_precision', cut.range=c(0,1.1), cut.num=11)

# 
# p1 <- loess_impute_plot(mtrx.melt, data.dt, z='lm_pearson_r', cut.range=c(0,1.1), cut.num=11)
# p2 <- loess_impute_plot(mtrx.melt, data.dt, z='gwas_pearson_r', cut.range=c(0,1.1), cut.num=11)
# 
# grid.arrange(p1, p2)
# 
# p1 <- loess_impute_plot(mtrx.melt, data.dt, z='lm_false_positives', cut.range=c(0,100), cut.num=11)
# p2 <- loess_impute_plot(mtrx.melt, data.dt, z='gwas_false_positives', cut.range=c(0,100), cut.num=11)
# 
# grid.arrange(p1, p2)
# 
# p1 <- loess_impute_plot(mtrx.melt, data.dt, z='lm_false_negatives', cut.range=c(0,120), cut.num=13)
# p2 <- loess_impute_plot(mtrx.melt, data.dt, z='gwas_false_negatives', cut.range=c(0,120), cut.num=13)
# 
# grid.arrange(p1, p2)


###### junk plots

# ggplot(data=dbl.dt) + 
#   geom_point(
#     aes(
#       x=num_snps_with_effect,
#       y=num_samples,
#       color=lm_percent_of_total_effect_detected), 
#     position='jitter') + 
#   scale_color_distiller(palette='Spectral', direction=1) +
#   theme_minimal()
# 
# ggplot(data=dbl.dt) + 
#   geom_point(
#     aes(
#       x=num_snps_with_effect,
#       y=num_samples,
#       color=gwas_specificity), 
#     position='jitter') + 
#   scale_color_distiller(palette='Spectral', direction=-1)
# 
# 
# ggplot(data=dbl.dt) + 
#   geom_point(
#     aes(
#       x=num_snps_with_effect,
#       y=num_samples,
#       color=gwas_specificity), 
#     position='jitter') + 
#   scale_color_distiller(palette='Spectral', direction=-1)
# 
# ggplot(
#   dbl.dt[, 
#     list(gwas_specificity=mean(gwas_specificity)), 
#     by=c('num_snps_with_effect','num_samples')],
#   aes(
#       x=num_snps_with_effect,
#       y=num_samples,
#       z=gwas_specificity)) + 
#   stat_contour(geom="polygon", binwidth = 0.01, aes(fill=..level..)) +
#   scale_fill_distiller(palette='Spectral', direction=1)
# 
# ggplot(
#   dbl.dt[, lapply(.SD, mean), by=c('num_snps_with_effect','num_samples')],
#   aes(
#       x=num_snps_with_effect,
#       y=num_samples,
#       z=gwas_pearson_r)) + 
#   stat_contour(geom="line", binwidth = 0.01, aes(color=..level..), size=1) +
#   scale_fill_distiller(palette='Spectral', direction=1)
# 
