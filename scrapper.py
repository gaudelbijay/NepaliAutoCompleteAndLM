from bs4 import BeautifulSoup
import requests

#TO-DO
'''
func scrapper:

Input: url from crawl_over_pages function, html_post_tag in post pages, starting_tag_number, 
ending_tag_number

output: .txt file containgng all the post 
'''

def scrapper(url, section='p', starting_p = 3, ending_p = -10,):
    response = requests.get(url)
    content = BeautifulSoup(response.content, "html.parser")
    content = content.findAll(section)
    post = ''
    for each_p in content[starting_p:ending_p]:
        post = post+each_p.text
    
    print('writing inot file....')
    with open('nepalitext.json', 'w') as fp: # will change this to txt file 
        json.dump(post, fp)
    print('completed writing into file ...')

#TO-DO

'''
func crawl_over_page:

Input: url to main pages, html_post_tag in main pages

main body: find all the links to the related html_post_tag

output: return list of all url for html_post_tag
'''