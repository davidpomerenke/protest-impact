import html2text
from readability import Document

text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.ignore_images = True
text_maker.ignore_emphasis = True
text_maker.ignore_tables = True


def html2text(html: str) -> tuple[str, str]:
    doc = Document(html)
    return doc.title(), text_maker.handle(doc.summary()).strip()
