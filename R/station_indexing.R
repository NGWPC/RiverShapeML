library(hydrofabric)

pois = read_parquet('data/station_ids.parquet') %>% 
  select(!!ID, lat, long) %>% 
  st_as_sf(coords = c("long", "lat"), crs = 4326) %>% 
  st_join(st_transform(vpu_boundaries, 4326))

bad  = filter(pois, is.na(VPUID))
ind = st_nearest_feature(bad, st_transform(vpu_boundaries, 4326))
bad$VPUID = vpu_boundaries$VPUID[ind]

pts = bind_rows(filter(pois, !is.na(VPUID)), bad)

sum(duplicated(pts$siteID)) == 0

pattern = '/Volumes/Transcend/ngen/CONUS-hydrofabric/01_reference/reference_{vpu}.gpkg'

v = unique(pts$VPUID)
ll = list()

for(i in 1:length(v)){
  
  x  = glue(pattern, vpu = v[i])
  fl = read_sf(x, "reference_flowline")
  
  mappedPOI = read_sf(x, paste0("POIs_", v[i])) %>% 
    select(COMID, siteID = Type_Gages) %>% 
    filter(!is.na(siteID)) %>% 
    st_transform(st_crs(fl)) %>% 
    rename_geometry("geometry") %>% 
    filter(siteID %in% pts$siteID)
  
  sum(duplicated(mappedPOI$COMID)) == 0
  sum(duplicated(mappedPOI$siteID)) == 0
  
  tmp = filter(pts, VPUID == v[i]) %>% 
    st_transform(st_crs(fl))
  
  need_to_map = filter(tmp, !siteID %in% mappedPOI$siteID) %>% 
    rename_geometry("geometry")
  
  mat = get_flowline_index(fl, 
                           need_to_map, 
                           search_radius = units::set_units(200, "m"))
  
  mat$siteID = need_to_map$siteID[mat$id]
  
  tmp2 = mat %>% 
    select(COMID, siteID) %>% 
    left_join(select(fl, COMID), by = "COMID") %>% 
    st_as_sf() %>% 
    mutate(geom = st_geometry(get_node(.))) %>% 
    rename_geometry("geometry") %>% 
    bind_rows(mappedPOI)


  sum(duplicated(tmp2$siteID)) == 0
  
  ll[[i]] = tmp2
  
  message(i)
}



xx = bind_rows(ll) %>% 
  st_join(st_transform(vpu_boundaries, 5070)) %>% 
  select(COMID, siteID, VPUID)

write_sf(xx, "data/site_index.gpkg", "sites")


# Part 2 ------------------------------------------------------------------

sites = read_sf("data/site_index.gpkg", "sites")
ll = list()

for(i in 1:length(v)){
  
  x  = glue(pattern, vpu = v[i])
  
  div = read_sf(x, "reference_catchment") %>% 
    mutate(VPU = v[i])
  
  ll[[i]] = filter(div, FEATUREID %in% sites$COMID)
  
  message(i)
}


write_sf(bind_rows(ll), "data/site_index.gpkg", "divides")




pts = read_sf("data/indexed_stations.gpkg")

sum(duplicated(xx$siteID))
