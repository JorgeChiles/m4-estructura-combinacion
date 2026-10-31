# exportar_m3.R
#
# Saca de los .rda del paquete Mcomp (Hyndman) tres CSV planos:
#
#   m3_series.csv       una fila por serie: id, periodo, tipo, n, h, m
#   m3_valores.csv      largo: id, parte (train|test), t, valor
#   m3_pronosticos.csv  ancho: id, metodo, f1 .. f18
#
# El segundo archivo son los pronosticos PUBLICADOS de los 24 metodos que
# compitieron en M3 ---no se reajusta nada, que es justo el punto: la validacion
# externa no vuelve a entrenar modelos, usa los errores que la competencia ya
# publico--- y entre ellos viene NAIVE2, que es el denominador del OWA.
#
# Uso:  Rscript exportar_m3.R <M3.rda> <M3Forecast.rda> <carpeta_salida>

a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 3)
e <- new.env()
load(a[1], envir = e)
load(a[2], envir = e)
M3 <- get("M3", e)
FC <- get("M3Forecast", e)
dir.create(a[3], showWarnings = FALSE, recursive = TRUE)

# m, el periodo estacional, es el de M4: 1 anual, 4 trimestral, 12 mensual.
# Las 174 series OTHER no declaran periodo (la ts viene con frecuencia 1), asi
# que se tratan con m = 1, igual que las anuales.
periodo_m <- c(YEARLY = 1L, QUARTERLY = 4L, MONTHLY = 12L, OTHER = 1L)

meta <- do.call(rbind, lapply(M3, function(s) data.frame(
  id = s$sn, periodo = s$period, tipo = s$type,
  n = length(s$x), h = length(s$xx), m = periodo_m[[s$period]],
  stringsAsFactors = FALSE)))
write.csv(meta, file.path(a[3], "m3_series.csv"), row.names = FALSE)

val <- do.call(rbind, lapply(M3, function(s) rbind(
  data.frame(id = s$sn, parte = "train", t = seq_along(s$x),
             valor = as.numeric(s$x), stringsAsFactors = FALSE),
  data.frame(id = s$sn, parte = "test", t = seq_along(s$xx),
             valor = as.numeric(s$xx), stringsAsFactors = FALSE))))
write.csv(val, file.path(a[3], "m3_valores.csv"), row.names = FALSE)

pron <- do.call(rbind, lapply(names(FC), function(nm) {
  M <- as.matrix(FC[[nm]])
  d <- as.data.frame(M)
  names(d) <- paste0("f", seq_len(ncol(M)))
  cbind(data.frame(id = rownames(M), metodo = nm, stringsAsFactors = FALSE), d)
}))
write.csv(pron, file.path(a[3], "m3_pronosticos.csv"), row.names = FALSE)

cat(nrow(meta), "series,", length(FC), "metodos,", nrow(pron), "filas de pronostico\n")
cat("periodos:", paste(names(table(meta$periodo)), table(meta$periodo),
                       collapse = "  "), "\n")
