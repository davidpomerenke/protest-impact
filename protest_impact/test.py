from datetime import date

from protest_impact.data.news.sources.nexis import search as nexis_search

result = nexis_search("burden of proof", date(2021, 2, 4))
print(result)
