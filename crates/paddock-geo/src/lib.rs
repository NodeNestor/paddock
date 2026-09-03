//! Offline place lookup: two decimal degrees in, a name a person recognises out.
//!
//! WHY. A photo's EXIF carries coordinates and nothing else, and a model handed
//! bare numbers does geography from memory. Measured on a real turn:
//! Qwen3.5-9B read `43.467448, 11.885127` as "the Piedmont region of northern
//! Italy" - it is Tuscany, 250 km away, and the nearest city is Arezzo at
//! 0.6 km. Everything else in that answer was sound; the one fact that came
//! from metadata was the one it invented. Resolving the point before it reaches
//! the prompt turns the guess into a fact.
//!
//! Why it is OFFLINE. Looking at your own photo must not tell anyone where you
//! were. A geocoding API would leak exactly the coordinate the user is asking
//! about, once per photo, silently - the thing this product says it does not
//! do. So the table ships in the binary: 170,756 places over ~1,000 people,
//! 3.5 MB packed down to 1.9 MB by zstd, decompressed once on first use.
//!
//! Why that TIER. GeoNames publishes the same data at three population floors.
//! Measured over 300 simulated rural photos: the coarse
//! cities15000 has a median error of 10.1 km and a p90 of 33 km, cities1000
//! 3.3 km and 8 km. A p90 of 33 km is a different valley - the exact mistake
//! this lookup exists to stop - so the extra 1.45 MB is bought deliberately.
//! The tiers agree wherever a photo sits near a large town; rural points are
//! where they part, and 77% of that sample got a different name.
//!
//! What it does not DO. This is nearest-populated-place, not point-in-polygon,
//! so the REGION is the region of the city we matched, not a boundary test on
//! the point itself. Near a border those differ. The phrasing everywhere
//! therefore attributes the region to the city ("in Arezzo (Tuscany, Italy)")
//! and never claims the point is inside a region on its own authority, and
//! [`Place::distance_km`] is always available so a caller can say how far.
//!
//! Data: GeoNames, CC by 4.0 - see THIRD-PARTY-NOTICES. Regenerate with
//! `python the geodata generator`.

use std::sync::OnceLock;

/// The packed table, zstd-compressed. The geodata generator writes it.
const PACKED: &[u8] = include_bytes!("../data/places.bin.zst");
const MAGIC: &[u8; 8] = b"PDGEO1\0\0";

/// A populated place near a coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    /// The place's own name, e.g. "Arezzo".
    pub city: String,
    /// First-level administrative division, e.g. "Tuscany". Empty when the
    /// source has no name for it (a handful of small territories).
    pub region: String,
    /// Country name, e.g. "Italy". Falls back to the ISO code if unnamed.
    pub country: String,
    /// Great-circle distance from the queried point to the place, in km.
    pub distance_km: f64,
    /// Compass direction from the place to the queried point ("NE"), so a
    /// caller can write "12 km NE of Arezzo".
    pub bearing: &'static str,
}

impl Place {
    /// One phrase, honest about how far away the name actually is. Inside
    /// ~3 km the distance is noise next to a town's own extent, so it reads
    /// "in X"; beyond that the number is stated rather than glossed over.
    pub fn describe(&self) -> String {
        let where_ = if self.region.is_empty() {
            self.country.clone()
        } else {
            format!("{}, {}", self.region, self.country)
        };
        if self.distance_km < 3.0 {
            format!("in {} ({where_})", self.city)
        } else {
            format!(
                "{:.0} km {} of {} ({where_})",
                self.distance_km, self.bearing, self.city
            )
        }
    }
}

struct City {
    lat: f64,
    lon: f64,
    region: u16,
    name: String,
}

struct Table {
    regions: Vec<(String, String)>,
    cities: Vec<City>,
}

fn table() -> Option<&'static Table> {
    static T: OnceLock<Option<Table>> = OnceLock::new();
    T.get_or_init(|| parse(PACKED)).as_ref()
}

/// Parse the packed blob. Returns `None` on any malformed byte rather than
/// panicking: a place name is a garnish, and a corrupt table must not take
/// down an extraction that would otherwise succeed.
fn parse(packed: &[u8]) -> Option<Table> {
    let raw = zstd::decode_all(packed).ok()?;
    let mut p = 0usize;
    let take = |p: &mut usize, n: usize| -> Option<&[u8]> {
        let s = raw.get(*p..p.checked_add(n)?)?;
        *p += n;
        Some(s)
    };
    if take(&mut p, 8)? != MAGIC {
        return None;
    }
    let u32at =
        |p: &mut usize| -> Option<u32> { Some(u32::from_le_bytes(take(p, 4)?.try_into().ok()?)) };

    let n_regions = u32at(&mut p)? as usize;
    let mut regions = Vec::with_capacity(n_regions);
    for _ in 0..n_regions {
        let len = take(&mut p, 1)?[0] as usize;
        let s = std::str::from_utf8(take(&mut p, len)?).ok()?;
        let (region, country) = s.split_once('\t').unwrap_or((s, ""));
        regions.push((region.to_owned(), country.to_owned()));
    }

    let n_cities = u32at(&mut p)? as usize;
    let mut cities = Vec::with_capacity(n_cities);
    for _ in 0..n_cities {
        let lat = i32::from_le_bytes(take(&mut p, 4)?.try_into().ok()?) as f64 / 1e4;
        let lon = i32::from_le_bytes(take(&mut p, 4)?.try_into().ok()?) as f64 / 1e4;
        let region = u16::from_le_bytes(take(&mut p, 2)?.try_into().ok()?);
        let len = take(&mut p, 1)?[0] as usize;
        let name = std::str::from_utf8(take(&mut p, len)?).ok()?.to_owned();
        cities.push(City {
            lat,
            lon,
            region,
            name,
        });
    }
    Some(Table { regions, cities })
}

fn haversine_km(a_lat: f64, a_lon: f64, b_lat: f64, b_lon: f64) -> f64 {
    const R: f64 = 6371.0088; // mean Earth radius, km
    let (p1, p2) = (a_lat.to_radians(), b_lat.to_radians());
    let dp = (b_lat - a_lat).to_radians();
    let dl = (b_lon - a_lon).to_radians();
    let h = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * R * h.sqrt().clamp(-1.0, 1.0).asin()
}

/// Initial great-circle bearing from one point to another, as a compass point.
fn compass(from_lat: f64, from_lon: f64, to_lat: f64, to_lon: f64) -> &'static str {
    let (p1, p2) = (from_lat.to_radians(), to_lat.to_radians());
    let dl = (to_lon - from_lon).to_radians();
    let y = dl.sin() * p2.cos();
    let x = p1.cos() * p2.sin() - p1.sin() * p2.cos() * dl.cos();
    let deg = (y.atan2(x).to_degrees() + 360.0) % 360.0;
    const POINTS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    POINTS[(((deg + 22.5) % 360.0) / 45.0) as usize % 8]
}

/// The populated place nearest `lat`/`lon`, or `None` when the coordinate is
/// not a real one or the table could not be read.
///
/// Linear over 171k rows with a cheap latitude reject first - around a
/// millisecond, far below the cost of the JPEG parse that produced the
/// coordinate, so there is no index to keep in sync and nothing to invalidate.
pub fn nearest(lat: f64, lon: f64) -> Option<Place> {
    if !lat.is_finite()
        || !lon.is_finite()
        || !(-90.0..=90.0).contains(&lat)
        || !(-180.0..=180.0).contains(&lon)
    {
        return None;
    }
    let t = table()?;
    let mut best: Option<(f64, &City)> = None;
    // Widening latitude bands: most queries settle inside the first, and a
    // point in an empty ocean still terminates because the last band covers
    // the whole table.
    for band in [2.0f64, 10.0, 45.0, 180.0] {
        for c in &t.cities {
            if (c.lat - lat).abs() > band {
                continue;
            }
            let d = haversine_km(lat, lon, c.lat, c.lon);
            if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                best = Some((d, c));
            }
        }
        // A hit well inside the band cannot be beaten by anything outside it
        // (1 degree of latitude is ~111 km, and longitude only ever narrows).
        if let Some((d, _)) = &best
            && *d < band * 111.0
        {
            break;
        }
    }
    let (distance_km, c) = best?;
    let (region, country) = t
        .regions
        .get(c.region as usize)
        .cloned()
        .unwrap_or_default();
    Some(Place {
        city: c.name.clone(),
        region,
        country,
        distance_km,
        bearing: compass(c.lat, c.lon, lat, lon),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The photo that started this (DSCN0010.jpg, exif-samples) - the one a
    /// model called Piedmont.
    #[test]
    fn resolves_the_tuscany_photo() {
        let p = nearest(43.467448, 11.885127).expect("a place");
        assert_eq!(p.city, "Arezzo");
        assert_eq!(p.region, "Tuscany");
        assert_eq!(p.country, "Italy");
        assert!(p.distance_km < 2.0, "{} km", p.distance_km);
        assert_eq!(p.describe(), "in Arezzo (Tuscany, Italy)");
    }

    /// Lat/lon the wrong way round must land somewhere else entirely - the
    /// case the labelled prompt line exists to prevent.
    #[test]
    fn swapped_coordinates_are_a_different_country() {
        let venice = nearest(45.44, 12.32).expect("venice");
        let swapped = nearest(12.32, 45.44).expect("swapped");
        assert_eq!(venice.country, "Italy");
        assert_ne!(swapped.country, "Italy");
    }

    #[test]
    fn distant_points_report_distance_and_bearing() {
        // mid-Atlantic: nothing is close, and the phrase must say so
        let p = nearest(30.0, -40.0).expect("a place");
        assert!(p.distance_km > 500.0, "{p:?}");
        assert!(p.describe().contains(" of "), "{}", p.describe());
    }

    #[test]
    fn rejects_impossible_coordinates() {
        assert!(nearest(91.0, 0.0).is_none());
        assert!(nearest(0.0, 181.0).is_none());
        assert!(nearest(f64::NAN, 0.0).is_none());
    }

    #[test]
    fn table_loads_and_is_the_expected_size() {
        let t = table().expect("table parses");
        assert_eq!(t.cities.len(), 170_756);
        assert!(t.regions.len() > 2_000);
    }

    #[test]
    fn compass_points_are_right_way_round() {
        assert_eq!(compass(0.0, 0.0, 1.0, 0.0), "N");
        assert_eq!(compass(0.0, 0.0, 0.0, 1.0), "E");
        assert_eq!(compass(0.0, 0.0, -1.0, 0.0), "S");
        assert_eq!(compass(0.0, 0.0, 0.0, -1.0), "W");
    }
}
