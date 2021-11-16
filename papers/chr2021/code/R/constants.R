current.language = 'Latin' # 'Icelandic' # 'Vedic'
data.dir.name = 'data'

if(current.language=='Vedic'){
}else if(current.language=='Icelandic'){
}else if(current.language=='Latin'){
  DATE_LOW = -300
  DATE_UP  = 1400
}
DATE_RANGE = DATE_UP - DATE_LOW


# Latin stopwords from Perseus
latinStopwords = c('ab', 'ac', 'ad', 'adhic', 'aliqui', 'aliquis', 'an', 'ante', 'apud', 'at', 'atque', 'aut',
                   'autem', 'cum', 'cur', 'de', 'deinde', 'dum', 'ego', 'enim', 'ergo', 'es', 'est', 'et', 
                   'etiam', 'etsi', 'ex', 'fio', 'haud', 'hic', 'iam', 'idem', 'igitur', 'ille', 'in', 'infra', 
                   'inter', 'interim', 'ipse', 'is', 'ita', 'magis', 'modo', 'mox', 'nam', 'ne', 'nec', 'necque', 
                   'neque', 'nisi', 'non', 'nos', 'o', 'ob', 'per', 'possum', 'post', 'pro', 'quae', 'quam', 
                   'quare', 'qui', 'quia', 'quicumque', 'quidem', 'quilibet', 'quis', 'quisnam', 'quisquam', 
                   'quisque', 'quisquis', 'quo', 'quoniam', 'sed', 'si', 'sic', 'sive', 'sub', 'sui', 'sum', 
                   'super', 'suus', 'tam', 'tamen', 'trans', 'tu', 'tum', 'ubi', 'uel', 'uero',
                   # extensions OH
                   'alius', 'hec', 'hiis', 'meus', 'nihil', 'noster', 'praeter', 'qui1', 'quidam', 'quoque', 'sive', 
                   'tamquam', 'tuus', 'uester'
                   )

# Latin: Ids of some authors
TextIdAlanus = 2
TextIdAlbertOfAix = 4
TextIdAlcuin = 5
TextIdAngilbert = 101
TextIdApuleius = 9
TextIdBeda = 14
TextIdBernardC = 15
TextIdCato = 20
TextIdCicero = 21
TextIdCommodianus = 25
TextIdEnnius = 119
TextIdErchempert = 121
TextIdFredegar = 128
TextIdLeoGreat = 48
TextIdLeoNaples = 47
TextIdLucretius = 51
TextIdMartial = 55
TextIdMaximianus = 144
TextIdNaevius = 147
TextIdNicoleO = 149
TextIdSenecaY = 75
TextIdVoragine = 94
TextIdVulgata = 166


# for the cross-validation
testedAuthors = c(
  # old
  TextIdNaevius, TextIdEnnius, TextIdCato,
  # classical
  TextIdCicero,TextIdSenecaY,TextIdApuleius,
  # vulgar
  TextIdCommodianus,TextIdLeoGreat,TextIdMaximianus,
  # transitional
  TextIdFredegar, TextIdAlcuin,TextIdErchempert,
  # medieval
  TextIdLeoNaples,TextIdBernardC,TextIdNicoleO
)


# see: https://en.wiktionary.org/wiki/Wiktionary:List_of_languages
RomanceLanguageCodes = c(
  'ast', # Asturian
  'ca',  # Catalan
  'dlm', # Dalmatian
  'es',
  'fr', 'fro', 'frm',
  'fur', # Friulian
  'gl', # Galician
  'it',
  'nrf', # Norman
  'oc', # Occitan
  'osp', # Old Spanish
  'pro', # Old Occitan
  'pt', # Portuguese
  'ro', # Romanian
  'rup', # Aromanian
  'scn', # Sicilian
  'vec' # Venetian
)