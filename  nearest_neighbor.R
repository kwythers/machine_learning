require(tidyverse)
require(data.table)
require(qqrepel)
require(dplyr)
require(ggplot2)
require(lubridate)
require(sp)
require(ggmap)
require(mapproj)
require(rspatial)
require(ggvoronoi)
require(rgeos)
require(geosphere)

# load site data
AllSwedishData_toMatchWeaSta <- read_csv("Desktop/sweden/AllSwedishData_toMatchWeaSta.dat")

# load climate staton data
Weather_Stations_Prec_20082012 <- read_csv("Desktop/sweden/Weather Stations Prec 20082012.dat")

PrercStations20122016 <- read_csv("Desktop/sweden/20122016PrercStations.dat")

PreciStations20132017 <- read_csv("Desktop/sweden/20132017PreciStations.dat")

Pre_by_station_Good19111915 <- read_csv("Desktop/sweden/Pre 19111915 by station Good.dat")

# sweden_precip_pts <- SpatialPoints(Weather_Stations_Prec_20082012[, 3:2], proj4string=CRS("+proj=longlat +datum=NAD83"))
# sweden_precip_pts <- SpatialPointsDataFrame(sweden_precip_pts, Weather_Stations_Prec_20082012)

sweden <- map_data("world") %>% filter(region == "Sweden")

sweden_map <- 
ggplot(data = Weather_Stations_Prec_20082012, aes(x = Longitude, y = Latitude)) +
#scale_fill_gradientn("Elevation", 
                       #colors=c("seagreen","darkgreen","green1","yellow","gold4", "sienna"),
                       #values=scales::rescale(c(-60,0,1000,2000,3000,4000))) + 
  #scale_color_gradientn("Elevation", 
                        #colors=c("seagreen","darkgreen","green1","yellow","gold4", "sienna"),
                        #values=scales::rescale(c(-60,0,1000,2000,3000,4000))) + 
  coord_quickmap() + 
  theme_minimal() + 
  theme(axis.text = element_blank(), 
        axis.title=element_blank()) 

sweden_map + geom_point(size = 0.5) + geom_path(data = sweden, aes(long, lat, group = group), color = "red") 

sweden_map +
  geom_voronoi(aes(fill = Elevation_m_asl), outline = sweden) + 
  geom_point(size = 0.5) +
  geom_point(data = all_sets, color = 'red', size = 1) + 
  geom_point(data = set2012, color = 'green', size = 1) +
  geom_point(data = set2016, color = 'yellow', size = 1) +
  geom_point(data = set2017, color = 'purple', size = 1) 

# set1
set1915 <- AllSwedishData_toMatchWeaSta %>% 
  filter(Year == '1915') %>% 
  dplyr::select(Longitude, Latitude) 

set2012 <- AllSwedishData_toMatchWeaSta %>% 
  filter(Year == '2012') %>% 
  dplyr::select(Longitude, Latitude)

set2016 <- AllSwedishData_toMatchWeaSta %>% 
  filter(Year == '2016') %>% 
  dplyr::select(Longitude, Latitude) 

set2017 <- AllSwedishData_toMatchWeaSta %>% 
  filter(Year == '2017') %>% 
  dplyr::select(Longitude, Latitude) 

# # for plotting all the data together
# all_sets <- bind_rows(set1915, set2012, set2016, set2017)

# set2
prec19111915 <- Pre_by_station_Good19111915 %>% 
  dplyr::select(Longitude, Latitude)

prec20082012 <- Weather_Stations_Prec_20082012 %>% 
  dplyr::select(Longitude, Latitude)

prec20122016 <- PrercStations20122016 %>% 
  rename(Latitude = lat, Longitude = long) %>% 
  dplyr::select(Longitude, Latitude)

prec20132017 <- PreciStations20132017 %>% 
  rename(Latitude = lat, Longitude = long) %>% 
  dplyr::select(Longitude, Latitude)

# convert set1 and set2 lat lons to spatial points
p1_1915 <- SpatialPoints(set1915)
p1_2012 <- SpatialPoints(set2012)
p1_2016 <- SpatialPoints(set2016)
p1_2017 <- SpatialPoints(set2017)

p2_19111915 <- SpatialPoints(prec19111915)
p2_20082012 <- SpatialPoints(prec20082012)
p2_20122016 <- SpatialPoints(prec20122016)
p2_20132017 <- SpatialPoints(prec20132017)


distp1_1915p2 <- function(p1_1915, p2_19111915) {
  dst1915 <- sqrt((p1_1915[1] - p2_19111915[1])^2 + (p1_1915[2] - p2_19111915[2])^2)
  return(dst1915)
}

distp1_2012p2 <- function(p1_2012, p2_20082012) {
  dst2012 <- sqrt((p1_2012[1] - p2_20082012[1])^2 + (p1_2012[2] - p2_20082012[2])^2)
  return(dst2012)
}

distp1_2016p2 <- function(p1_2016, p2_20122016) {
  dst2016 <- sqrt((p1_2016[1] - p2_20122016[1])^2 + (p1_2016[2] - p2_20122016[2])^2)
  return(dst2016)
}

distp1_2017p2 <- function(p1_2017, p2_20132017) {
  dst2017 <- sqrt((p1_2017[1] - p2_20132017[1])^2 + (p1_2017[2] - p2_20132017[2])^2)
  return(dst2017)
}

# get min distance for each climate station year group 
dist2b1915 <- function(y) which.min(apply(prec19111915, 1, function(x) min(distp1_1915p2(x,y))))
dist1915 <- tibble(apply(set1915, 1, dist2b1915))

dist2b2012 <- function(y) which.min(apply(prec20082012, 1, function(x) min(distp1_2012p2(x,y))))
dist2012 <- tibble(apply(set2012, 1, dist2b2012))

dist2b2016 <- function(y) which.min(apply(prec20122016, 1, function(x) min(distp1_2016p2(x,y))))
dist2016 <- tibble(apply(set2016, 1, dist2b2016))

dist2b2017 <- function(y) which.min(apply(prec20132017, 1, function(x) min(distp1_2017p2(x,y))))
dist2017 <- tibble(apply(set2017, 1, dist2b2017))

# find nearest year group climate station for each set of site coords 
nearest_prec1915 <- prec19111915[apply(set1915, 1, dist2b1915),] %>% 
  add_column(year = 1915) %>% 
  rename(Latitude1 = Latitude, Longitude1 = Longitude)
nearest_prec1915 <- left_join(nearest_prec1915, Pre_by_station_Good19111915, by = c("Longitude1" = "Longitude", "Latitude1" = "Latitude"))
nearest_prec1915 <- nearest_prec1915 %>% rename(map = Mean_Sum_MAP, elevation = Elevation_m_asl) 
nearest_prec1915 <- nearest_prec1915 %>% dplyr::select(Longitude1, Latitude1, year, elevation, map) %>% 
  bind_cols(dist1915) %>% 
  rename(km = `apply(set1915, 1, dist2b1915)`)

nearest_prec2012 <- prec20082012[apply(set2012, 1, dist2b2012),] %>% 
  add_column(year = 2012)%>% 
  rename(Latitude1 = Latitude, Longitude1 = Longitude)
nearest_prec2012 <- left_join(nearest_prec2012, Weather_Stations_Prec_20082012, by = c("Longitude1" = "Longitude", "Latitude1" = "Latitude"))
nearest_prec2012 <- nearest_prec2012 %>% rename(map = MAPmean, elevation = Elevation_m_asl) 
nearest_prec2012 <- nearest_prec2012 %>% dplyr::select(Longitude1, Latitude1, year, elevation, map) %>% 
  bind_cols(dist2012) %>% 
  rename(km = `apply(set2012, 1, dist2b2012)`)

nearest_prec2016 <- prec20122016[apply(set2016, 1, dist2b2016),] %>% 
  add_column(year = 2016)%>% 
  rename(Latitude1 = Latitude, Longitude1 = Longitude)
nearest_prec2016 <- left_join(nearest_prec2016, PrercStations20122016, by = c("Longitude1" = "long", "Latitude1" = "lat")) 
nearest_prec2016 <- nearest_prec2016 %>% rename(map = Mean_Mean_MAP, elevation = alt) %>% 
  mutate(map = map *12)
nearest_prec2016 <- nearest_prec2016 %>% dplyr::select(Longitude1, Latitude1, year, elevation, map) %>% 
  bind_cols(dist2016) %>% 
  rename(km = `apply(set2016, 1, dist2b2016)`)

nearest_prec2017 <- prec20132017[apply(set2017, 1, dist2b2017),] %>% 
  add_column(year = 2017)%>% 
  rename(Latitude1 = Latitude, Longitude1 = Longitude)
nearest_prec2017 <- left_join(nearest_prec2017, PreciStations20132017, by = c("Longitude1" = "long", "Latitude1" = "lat")) 
nearest_prec2017 <- nearest_prec2017 %>% rename(map = Mean_Mean_MAP, elevation = alt) %>% 
  mutate(map = map *12)
nearest_prec2017 <- nearest_prec2017 %>% dplyr::select(Longitude1, Latitude1, year, elevation, map) %>% 
  bind_cols(dist2017) %>% 
  rename(km = `apply(set2017, 1, dist2b2017)`)

nearest_prec1915 <- bind_cols(set1915, nearest_prec1915) 
nearest_prec2012 <- bind_cols(set2012, nearest_prec2012)
nearest_prec2016 <- bind_cols(set2016, nearest_prec2016)
nearest_prec2017 <- bind_cols(set2017, nearest_prec2017)

all_nearest_prec <- bind_rows(nearest_prec1915, nearest_prec2012, nearest_prec2016, nearest_prec2017) %>% 
  dplyr::select(-Latitude2, -Longitude2)

site_prec <- full_join(AllSwedishData_toMatchWeaSta, all_nearest_prec, by = c("Longitude" = "Longitude", "Latitude" = "Latitude")) %>% 
  dplyr::select(-Latitude2, -Longitude2)

write_csv(all_nearest_prec, '~/Desktop/sweden/nearest_prec.csv')


