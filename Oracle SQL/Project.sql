--Q.1)	Extract the Month, Day & Year In three different columns in Calendar Table. If Table not created please create the table based on the file received.
select SUBSTR(c_date,1,2) "Date",
extract (month from c_date) "Month",
SUBSTR(c_date,7,2) "Year" from calendars;

--Q.2)	Create a new column in Cutomer Table as Full Name and let it have values from Prefix, First Name & Last Name.
alter table customers ADD FULL_NAME varchar2(40);
update customers set full_name = prefix || firstname || ' ' || lastname;
Select Prefix, FirstName, LastName, Full_name from Customers;

--Q.3)	Write a query to find out the number of customer who are married.
select count(*) from customers where maritalstatus='M';

--Q.4)	Replace the ($ , ) values from Annual Income and put the values in a new column that is Salary as numeric field.	
select annualincome,replace(replace(annualincome,'$'),',') as salary from customers;

--Q.5)	Write a query to find out how many customers have 0 kids.
SELECT FIRSTNAME,totalchildren FROM CUSTOMERS;
select count(*) from customers where totalchildren=0;

/*Q.6)	Give Bonus to the following customer occupation. For other O
Professional	50000
Clerical	10000
Management	25000
Manual	2000 */
select case  when occupation='Professional' then 50000  
             when occupation='Clerical' then 10000
             when occupation='Management' then 25000
             when occupation='Manual' then 2000
       else 0
end as bonus, annualincome,occupation  
from customers c;

--Q.7)	Give me a count of customers who have their own property. 
select count(*) from customers where homeowner = 'Y';

--Q.8)	Write a SQL Query to find out the Customer Last Name starts with ‘RA’ & FIRST Name ending with ‘DA’and ensure there is no duplicate records seen when the output is displayed.
select firstname,lastname from customers
where firstname like '%DA' AND LASTNAME like 'RA%'
group by (lastname,firstname) having count(*) <=1;

--Q.9)	Write a query to display the sales for the order date 03/21/2017 for product key 540.
select * from sales
where orderdate='21-MAR-2017' and productkey='540';

--Q.10)	Write a SQL Query to increase the cost of products by 18% and round the data to the nearest number.
select productcost,round(productcost+(0.18*productcost)) as new_productcost 
from products;

--Q.11)	Adventure work Head of sales would like to find out the cost difference between productcost and productprice.
select productprice-productcost as cost_difference
from products;

--Q.12)	Write a SQL Query to find out, which products were not, returned (Use tables Product & Returns) solve the query without ‘not in’ function.
select productname AS NOT_RETURNED from products where productkey in
    (select productkey from products minus
        select productkey from returns);

--Q.13)	Write a query to find out which customer has placed most number of sales.
SELECT Full_name,S.orderquantity 
           FROM CUSTOMERS C 
           LEFT OUTER JOIN SALES S
           ON C.CUSTOMERKEY=S.CUSTOMERKEY
    where s.orderquantity in (select max(orderquantity) from sales); 

--ANOTHER ALTERNATIVE ,GETTING ERROR!
/*SELECT full_name, COUNT(*) AS orderquantity 
FROM customers
JOIN sales ON customers.customerkey = sales.customerkey
GROUP BY full_name
HAVING COUNT(*) = (SELECT MAX(c) 
                   FROM (SELECT COUNT(*) AS c 
                         FROM customers 
                         JOIN sales ON customers.customerkey = sales.customerkey 
                         GROUP BY full_name) AS orderquantity)); */
___________________________________________________________________________-
--Q.14)	Write a SQL Query to find out the products returned for Region Germany.
select R.productkey,
       P.productname,
       T.Region
from returns R 
left outer join products P
on R.productkey=P.productkey

left outer join territories T
on R.territorykey=T.salesterritorykey
 where T.Region='Germany'
group by R.productkey,P.productname,T.Region;

--Q.15)	Adventure works have decided to change the product colour for a few of their products along which with their product size. Following is the information.
/*_COLOR_	       _New color_
RED             	BLACK
NA	                BLUE
MULTI               YELLOW

_PRODUCT SIZE_    	_NEW SIZE_
0	                 LARGE
XL	                 MEDIUM
ALL OTHERS         	 SMALL
*/
select case 
       when productcolor='Red' then 'Black'
       when productcolor='NA' then 'Blue'
       when productcolor='Multi' then 'Yellow'
    else productcolor
    end as n_color,productcolor,
       
       case 
        when productsize='0' then 'LARGE'
        when productsize='XL' then 'MEDIUM'
    else 'SMALL'
    end as n_size,productsize
from products;

--Q.16)	Write a sql query to find out the customers that have at least one sale from Northwest region of America.
select C.customerkey,
       C.firstname,
       C.lastname,
       S.territorykey, region,
COUNT(C.CUSTOMERKEY) AS SALES
from customers C
left outer join sales S
on C.customerkey=S.customerkey

left outer join territories T
on S.territorykey=T.salesterritorykey
where region='Northwest' 
group by  C.customerkey,C.firstname,C.lastname,s.territorykey,region
order by sales desc;

--Q.17)	Write a SQL Query to find out which customer has more than one order quantity.
select  c.FIRSTNAME,c.LASTNAME,c.customerkey,s.orderquantity
from customers c
left outer join sales s
on c.customerkey=s.customerkey 
where orderquantity > 1 
group by c.firstname,c.lastname,c.customerkey, s.orderquantity;

--Q.18)	Write a query to find out in which region the following sub category Road Bikes, Mountain Frames are sold and by which customer. Use CTE
     with res as 
(select c.customerkey as custkey,
        c.firstname as fname,
        c.lastname as lname
         from customers c),
vi as
(select    s.customerkey as customerkey,
        s.productkey as productkey,
        t.region as region,
        t.salesterritorykey,
        s.territorykey
from sales s 
inner join territories t
on s.territorykey = t.salesterritorykey),
x as 
(select ps.productkey as prodkey,
		ps.productsubcategorykey as productsubcategorykey
		from products ps) ,
 v as 
(select p.productsubcategorykey as pskey,
		p.subcategoryname as name
from productsubcategory p)
select  res.fname,res.lname,vi.region, res.custkey, vi.customerkey, vi.productkey ,x.prodkey, x.productsubcategorykey, v.name
	from res 
	inner join vi
on res.custkey = vi.customerkey
inner join  x
on vi.productkey = x.prodkey
inner join v
on x.productsubcategorykey = v.pskey
where v.name like '%Mountain Bikes%' or v.name like '%Road Bike%';

--Q.19)	Write a SQL Query to find out which products were returned.
select productname as RETURNED_PRODUCTS from products 
where productkey in (select productkey from returns);

--Q.20)	Write a query to add a new column in customers table as username and get the values from email field. Fetch all the values before @ symbol. 
--Update the new field with the values populated your query.
ALTER TABLE CUSTOMERS ADD USERNAME VARCHAR2(40);
UPDATE CUSTOMERS SET USERNAME=SUBSTR(EMAILADDRESS,0,INSTR(EMAILADDRESS,'@')-1);
SELECT USERNAME, EMAILADDRESS FROM CUSTOMERS ;

/*Q.21)	Write a SQL Query to find get a report for the following 
a)	List of all customers
b)	Sales done by each customer
c)	Product owned by each customer
d)	Name of the Product Sub category
e)	Products, which were returned.
*/
WITH A_CTE AS 
(SELECT T1.CUSTOMERKEY AS cust_customerkey,
       T1.FULL_NAME AS CUST_FULL_NAME
FROM CUSTOMERS T1),
B AS
(SELECT T2.PRODUCTKEY AS PRODUCTKEY_SALES,
        T2.CUSTOMERKEY AS SALES_CUSTOMERKEY
FROM SALES T2),
C AS
(SELECT T3.PRODUCTKEY AS PRODUCTKEY_PRODS,
        T3.PRODUCTSUBCATEGORYKEY AS PROD_SUB_CAT_KEY_PRODS,
        T3.PRODUCTNAME AS PRODUCT_NAME_PRODS,
        T4.PRODUCTKEY AS PRODUCTKEY_RETURN
FROM PRODUCTS T3
LEFT OUTER JOIN RETURNS T4
ON T3.PRODUCTKEY=T4.PRODUCTKEY),
D AS
(SELECT T5.PRODUCTSUBCATEGORYKEY AS PROD_SUB_CAT_KEY_PRODSUBCAT,
        T5.SUBCATEGORYNAME AS PROD_SUB_CAT_NAME
FROM productsubcategory T5)
SELECT DISTINCT CUST_FULL_NAME,PRODUCT_NAME_PRODS,PRODUCTKEY_RETURN,PROD_SUB_CAT_NAME FROM A_CTE
LEFT OUTER JOIN B
ON A_CTE.cust_customerkey = B.SALES_CUSTOMERKEY
LEFT OUTER JOIN C
ON B.PRODUCTKEY_SALES = C.PRODUCTKEY_PRODS
LEFT OUTER JOIN D
ON C.PROD_SUB_CAT_KEY_PRODS = D.PROD_SUB_CAT_KEY_PRODSUBCAT;

--Q.22)	Write a SQL Query using Sub-select to get the count of all table.
select count(*)  from
 (SELECT T1.CUSTOMERKEY AS cust_customerkey,
         T1.FULL_NAME AS CUST_FULL_NAME
FROM CUSTOMERS T1) A
LEFT OUTER JOIN
 (SELECT T2.PRODUCTKEY AS PRODUCTKEY_SALES,
         T2.CUSTOMERKEY AS SALES_CUSTOMERKEY
FROM SALES T2) B
ON A.cust_customerkey = B.SALES_CUSTOMERKEY;

--Q.23)	Write a SQL Query to find out which customer has 3rd highest salary using common table expression.
WITH A_CTE AS
(SELECT full_name,annualincome,
DENSE_RANK() OVER (ORDER BY annualincome Desc) AS nth
FROM customers)
SELECT * FROM A_CTE
WHERE nth=3;

--Q.24)	Write a query to replace the Gender value NA to Null.
select firstname,gender,
case when gender = 'NA' then null
else gender
end as gen
from customers;











































































