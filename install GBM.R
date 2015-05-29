# reinstall gbm to fix memory leak issues
library(devtools)
install_github("gbm-developers/gbm")  # though this is the newer version, it has serious memory leak issues
install_github("harrysouthworth/gbm")  # use this version to fix the memory leak issue

sourceDir <- function(path, trace = TRUE, ...) {
  for (nm in list.files(path, pattern = "[.][RrSsQq]$")) {
    if(trace) cat(nm,":")
    source(file.path(path, nm), ...)
    if(trace) cat("\n")
  }
}

sourceDir('C:/Program Files/R/R-3.1.3/library/gbm/R')
