import os

meta_data = {
        ("accounts_profiles", "https://d3a0d0y2hgofx6.cloudfront.net/openapi/en-us/profiles/3-0/openapi.yaml"),
        ("accounts_manager_accounts", "https://dtrnk0o2zy01c.cloudfront.net/openapi/en-us/dest/ManagerAccount_prod_3p.json"),
        ("accounts_ads_accounts", "https://dtrnk0o2zy01c.cloudfront.net/openapi/en-us/dest/AdvertisingAccounts_prod_3p.json"),
        ("accounts_portfolios", "https://d3a0d0y2hgofx6.cloudfront.net/openapi/en-us/portfolios/openapi.yaml"),
        ("accounts_billings", "https://dtrnk0o2zy01c.cloudfront.net/openapi/en-us/dest/AdvertisingBilling_prod_3p.json"),
        ("accounts_account_budgets", "https://dtrnk0o2zy01c.cloudfront.net/openapi/en-us/dest/Advertisers_prod_3p.json"),
        ("sponsored_brands_v4", "https://d3a0d0y2hgofx6.cloudfront.net/openapi/en-us/sponsored-brands/4-0/openapi.json"),
        ("sponsored_brands_v3", "https://d3a0d0y2hgofx6.cloudfront.net/openapi/en-us/sponsored-brands/3-0/openapi.yaml")
        }

if __name__ == '__main__':
    for item in meta_data:
        file_name = item[0]
        url = item[1]
        file_type = url.split('/')[-1].split('.')[1]
        os.system(f'cd data && wget {url} -O {file_name}.{file_type}'.format(url, file_name, file_type))
