library(hydrofabric)

pois = read_parquet('data/station_ids.parquet') %>% 
  select(!!ID, lat, long) %>% 
  st_as_sf(coords = c("long", "lat"), crs = 4326) %>% 
  st_join(st_transform(vpu_boundaries, 4326))

bad  = filter(pois, is.na(VPUID))
ind = st_nearest_feature(bad, st_transform(vpu_boundaries, 4326))
bad$VPUID = vpu_boundaries$VPUID[ind]

pts = bind_rows(filter(pois, !is.na(VPUID)), bad)

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
    rename_geometry("geometry")
  
  tmp = filter(pts, VPUID == v[i]) %>% 
    st_transform(st_crs(fl))
  
  left_over = filter(tmp, !siteID %in% mappedPOI$siteID) %>% 
    rename_geometry("geometry")
  
  good_to_go = filter(mappedPOI, siteID %in% tmp$siteID)

  mat = get_flowline_index(fl, 
                           left_over, 
                           search_radius = units::set_units(200, "m"))
  
  mat$siteID = tmp$siteID[mat$id]
  
  ll[[i]] = mat %>% 
    select(COMID, siteID) %>% 
    left_join(select(fl, COMID), by = "COMID") %>% 
    st_as_sf() %>% 
    mutate(geom = st_geometry(get_node(.))) %>% 
    rename_geometry("geometry") %>% 
    bind_rows(good_to_go)
  
  message(i)
}



xx = bind_rows(ll)

write_sf(xx, "data/indexed_stations.gpkg")
