# coding: utf-8

categories = {}
category_tsv_path = ARGV.shift

open(category_tsv_path, 'r').each_line do |line|
  name, id = line.chomp.split("\t")
  categories[id] = name
end

puts ["product_id", "予測カテゴリ", "正解カテゴリ", "URL"].join("\t")

while log_line = gets
  if m = log_line.match(/product_id (.+?) - category predicted: (.+?) actual: (.+?)$/)
    product   = m[1]
    predicted = categories[m[2]]
    actual    = categories[m[3]]
    url       = "http://mall.tab.do/ja/products/#{product}"
    puts [product, predicted, actual, url].join("\t")
  end
end
