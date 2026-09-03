//! One module per search engine. Each exposes the same
//! `search(cfg, opts, query) -> Vec<Hit>` and is free to be as unlike the
//! others as its API is: the normalizing happens in `Hit` and in
//! `http::finish`, not by pretending five different engines are one.
//!
//! Every module's header says which of that provider's features we reach and
//! - just as important - which we deliberately do not send and why. A knob
//!   with no caller-side signal is dead config, and a knob we cannot verify
//!   against the provider's docs is a 400 waiting to happen.

pub(crate) mod brave;
pub(crate) mod exa;
pub(crate) mod firecrawl;
pub(crate) mod perplexity;
pub(crate) mod tavily;

/// ISO-3166-1 alpha-2 -> the English name search engines expect to read.
///
/// This exists because Tavily's `country` is a name ("united states"), not a
/// code, and both API dialects hand us a code - sending "us" there is
/// silently wrong, and against a strict enum it is a failed search. Firecrawl
/// reuses it for the country part of its free-form `location`.
///
/// The list is the 193 UN member states plus the two observers, which is
/// exactly the set Tavily documents ("195+ country options"). A territory that
/// isn't on it returns None, and the caller then sends no country at all -
/// better than guessing a spelling the provider will reject.
///
/// Spellings are BEST-EFFORT and deliberately conservative: Tavily's enum uses
/// older short names ("czech republic", "turkey", "congo") rather than the
/// current ISO ones ("czechia", "türkiye"), so those follow Tavily. Nobody can
/// promise all 195 match, which is why a provider rejecting the field costs
/// the geo hint and not the search - see `http::retryable`.
pub(crate) fn country_name(code: &str) -> Option<&'static str> {
    NAMES.iter().find(|(c, _)| *c == code).map(|(_, n)| *n)
}

#[rustfmt::skip]
const NAMES: [(&str, &str); 195] = [
    ("ad", "andorra"),
    ("ae", "united arab emirates"),
    ("af", "afghanistan"),
    ("ag", "antigua and barbuda"),
    ("al", "albania"),
    ("am", "armenia"),
    ("ao", "angola"),
    ("ar", "argentina"),
    ("at", "austria"),
    ("au", "australia"),
    ("az", "azerbaijan"),
    ("ba", "bosnia and herzegovina"),
    ("bb", "barbados"),
    ("bd", "bangladesh"),
    ("be", "belgium"),
    ("bf", "burkina faso"),
    ("bg", "bulgaria"),
    ("bh", "bahrain"),
    ("bi", "burundi"),
    ("bj", "benin"),
    ("bn", "brunei"),
    ("bo", "bolivia"),
    ("br", "brazil"),
    ("bs", "bahamas"),
    ("bt", "bhutan"),
    ("bw", "botswana"),
    ("by", "belarus"),
    ("bz", "belize"),
    ("ca", "canada"),
    ("cd", "democratic republic of the congo"),
    ("cf", "central african republic"),
    ("cg", "congo"),
    ("ch", "switzerland"),
    ("ci", "cote d'ivoire"),
    ("cl", "chile"),
    ("cm", "cameroon"),
    ("cn", "china"),
    ("co", "colombia"),
    ("cr", "costa rica"),
    ("cu", "cuba"),
    ("cv", "cape verde"),
    ("cy", "cyprus"),
    ("cz", "czech republic"),
    ("de", "germany"),
    ("dj", "djibouti"),
    ("dk", "denmark"),
    ("dm", "dominica"),
    ("do", "dominican republic"),
    ("dz", "algeria"),
    ("ec", "ecuador"),
    ("ee", "estonia"),
    ("eg", "egypt"),
    ("er", "eritrea"),
    ("es", "spain"),
    ("et", "ethiopia"),
    ("fi", "finland"),
    ("fj", "fiji"),
    ("fm", "micronesia"),
    ("fr", "france"),
    ("ga", "gabon"),
    ("gb", "united kingdom"),
    ("gd", "grenada"),
    ("ge", "georgia"),
    ("gh", "ghana"),
    ("gm", "gambia"),
    ("gn", "guinea"),
    ("gq", "equatorial guinea"),
    ("gr", "greece"),
    ("gt", "guatemala"),
    ("gw", "guinea-bissau"),
    ("gy", "guyana"),
    ("hn", "honduras"),
    ("hr", "croatia"),
    ("ht", "haiti"),
    ("hu", "hungary"),
    ("id", "indonesia"),
    ("ie", "ireland"),
    ("il", "israel"),
    ("in", "india"),
    ("iq", "iraq"),
    ("ir", "iran"),
    ("is", "iceland"),
    ("it", "italy"),
    ("jm", "jamaica"),
    ("jo", "jordan"),
    ("jp", "japan"),
    ("ke", "kenya"),
    ("kg", "kyrgyzstan"),
    ("kh", "cambodia"),
    ("ki", "kiribati"),
    ("km", "comoros"),
    ("kn", "saint kitts and nevis"),
    ("kp", "north korea"),
    ("kr", "south korea"),
    ("kw", "kuwait"),
    ("kz", "kazakhstan"),
    ("la", "laos"),
    ("lb", "lebanon"),
    ("lc", "saint lucia"),
    ("li", "liechtenstein"),
    ("lk", "sri lanka"),
    ("lr", "liberia"),
    ("ls", "lesotho"),
    ("lt", "lithuania"),
    ("lu", "luxembourg"),
    ("lv", "latvia"),
    ("ly", "libya"),
    ("ma", "morocco"),
    ("mc", "monaco"),
    ("md", "moldova"),
    ("me", "montenegro"),
    ("mg", "madagascar"),
    ("mh", "marshall islands"),
    ("mk", "north macedonia"),
    ("ml", "mali"),
    ("mm", "myanmar"),
    ("mn", "mongolia"),
    ("mr", "mauritania"),
    ("mt", "malta"),
    ("mu", "mauritius"),
    ("mv", "maldives"),
    ("mw", "malawi"),
    ("mx", "mexico"),
    ("my", "malaysia"),
    ("mz", "mozambique"),
    ("na", "namibia"),
    ("ne", "niger"),
    ("ng", "nigeria"),
    ("ni", "nicaragua"),
    ("nl", "netherlands"),
    ("no", "norway"),
    ("np", "nepal"),
    ("nr", "nauru"),
    ("nz", "new zealand"),
    ("om", "oman"),
    ("pa", "panama"),
    ("pe", "peru"),
    ("pg", "papua new guinea"),
    ("ph", "philippines"),
    ("pk", "pakistan"),
    ("pl", "poland"),
    ("ps", "palestine"),
    ("pt", "portugal"),
    ("pw", "palau"),
    ("py", "paraguay"),
    ("qa", "qatar"),
    ("ro", "romania"),
    ("rs", "serbia"),
    ("ru", "russia"),
    ("rw", "rwanda"),
    ("sa", "saudi arabia"),
    ("sb", "solomon islands"),
    ("sc", "seychelles"),
    ("sd", "sudan"),
    ("se", "sweden"),
    ("sg", "singapore"),
    ("si", "slovenia"),
    ("sk", "slovakia"),
    ("sl", "sierra leone"),
    ("sm", "san marino"),
    ("sn", "senegal"),
    ("so", "somalia"),
    ("sr", "suriname"),
    ("ss", "south sudan"),
    ("st", "são tomé and príncipe"),
    ("sv", "el salvador"),
    ("sy", "syria"),
    ("sz", "eswatini"),
    ("td", "chad"),
    ("tg", "togo"),
    ("th", "thailand"),
    ("tj", "tajikistan"),
    ("tl", "timor-leste"),
    ("tm", "turkmenistan"),
    ("tn", "tunisia"),
    ("to", "tonga"),
    ("tr", "turkey"),
    ("tt", "trinidad and tobago"),
    ("tv", "tuvalu"),
    ("tz", "tanzania"),
    ("ua", "ukraine"),
    ("ug", "uganda"),
    ("us", "united states"),
    ("uy", "uruguay"),
    ("uz", "uzbekistan"),
    ("va", "vatican city"),
    ("vc", "saint vincent and the grenadines"),
    ("ve", "venezuela"),
    ("vn", "vietnam"),
    ("vu", "vanuatu"),
    ("ws", "samoa"),
    ("ye", "yemen"),
    ("za", "south africa"),
    ("zm", "zambia"),
    ("zw", "zimbabwe"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn country_names_are_sorted_unique_and_engine_shaped() {
        // sorted + unique so the lookup stays trivially auditable
        for w in NAMES.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "{} is out of order before {}",
                w[0].0,
                w[1].0
            );
        }
        for (code, name) in NAMES {
            assert_eq!(code.len(), 2, "{code} is not an alpha-2 code");
            assert!(
                code.chars().all(|c| c.is_ascii_lowercase()),
                "{code} is not lower-case"
            );
            // Tavily matches on lower-case common names; a title-cased or
            // ISO long-form name ("Korea (Republic of)") would silently miss
            assert_eq!(name, name.to_lowercase(), "{name} is not lower-case");
            assert!(
                !name.contains('('),
                "{name} is an ISO long-form, not a common name"
            );
        }
        assert_eq!(country_name("us"), Some("united states"));
        assert_eq!(country_name("gb"), Some("united kingdom"));
        assert_eq!(country_name("se"), Some("sweden"));
        assert_eq!(country_name("kr"), Some("south korea"));
        // a territory outside the documented set sends no country at all
        assert_eq!(country_name("hk"), None);
        assert_eq!(country_name("zz"), None);
    }
}
