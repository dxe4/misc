import csvtools
import json


let input = open("stories.json")

type Article = object
  title: string
  score, time: int

var articles: seq[Article]
var skipped: int = 0

for line in input.lines:
    let jsonNode = parseJson(line)
    var article: Article
    if jsonNode["body"].hasKey("title"):
        article = Article(
            title: jsonNode["body"]["title"].getStr(),
            time: jsonNode["body"]["time"].getInt(),
            score: jsonNode["body"]["score"].getInt(),
        )
        articles.add(article)
    else:
        skipped += 1

articles.writeToCsv("stories.csv")

input.close()

echo skipped