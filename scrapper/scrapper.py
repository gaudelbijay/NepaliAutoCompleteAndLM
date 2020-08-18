import os
import bs4 as BeautifulSoup
import requests


'''
func crawl_over_page:

Input: url to main pages, html_post_tag in main pages

main body: find all the links to the related html_post_tag

output: return list of all url for html_post_tag
'''

def crawl_over_page(url, section='h2', starting_section=0, ending_section=-1):
    response = requests.get(url)
    content = BeautifulSoup(response.content, 'html.parser')
    content = content.findAll(section)
    post_url = [each_content.a['href'] for each_content in content[starting_section:ending_section]]
    return post_url


'''
func scrapper:

Input: url from crawl_over_pages function, html_post_tag in post pages, starting_tag_number, 
ending_tag_number

output: .txt file containgng all the post 
'''

def scrapper(url, starting_p=3, ending_p=-10, section='p'):
    urls = crawl_over_page(url)
    post = ''
    for url in urls:
        response = requests.get(url)
        content = BeautifulSoup(response.content, "html.parser")
        content = content.findAll(section)

        for each_p in content[starting_p:ending_p]:
            post = post + each_p.text

        with open('stories.txt', encoding='utf-8', mode='w') as f:
            f.write(post)


# run the code
if __name__ == '__main__':
    url = 'http://inepal.org/nepalistories/tag/read-nepali-stories-online/?fbclid=IwAR15RcMQqjA6LdJ5wyGKwiC_VdmYu-4FP1X1wBoeKfw0uOvuYEkgg-xJhgo'

    scrapper(url)



