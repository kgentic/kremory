# Spike: Single-call LLM extraction — quality samples

Generated: 2026-04-09T06:23:28Z

## Mock Interview

**Input (first 200 chars):**
```
Hi, I'm Ria. Thank you for coming to interview with us today. Let's just go ahead and get started with a quick introduction. Could you tell me a little bit about yourself? Yeah, absolutely. First of a
```

**Raw LLM output (first 500 chars):**
```
{"entities":[{"name":"Ria","label":"Person"},{"name":"Morocco","label":"Location"},{"name":"Boston","label":"Location"},{"name":"Northeastern University","label":"Organisation"},{"name":"Boston Consulting Group","label":"Organisation"},{"name":"Power Plant","label":"Organisation"},{"name":"Amazon Robotics","label":"Organisation"},{"name":"Amazon","label":"Organisation"},{"name":"BCG","label":"Organisation"},{"name":"South Korea","label":"Location"},{"name":"France","label":"Location"},{"name":"J
```

**All extracted entities (14):**
- Ria
- Morocco
- Boston
- Northeastern University
- Boston Consulting Group
- Power Plant
- Amazon Robotics
- Amazon
- BCG
- South Korea
- France
- Japan
- C-suite executives
- education

**All relationships (0):**

## Medical Consult

**Input (first 200 chars):**
```
Dr. Yemi Adebayo: Good morning, Mrs. Fortunato. I've reviewed your file from St. Catherine's General Hospital. Let's go through what's been happening since your last visit.

Mrs. Rosa Fortunato: Morni
```

**Raw LLM output (first 500 chars):**
```
{
  "entities": [
    {"name": "Dr. Yemi Adebayo", "label": "Person"},
    {"name": "Mrs. Rosa Fortunato", "label": "Person"},
    {"name": "St. Catherine's General Hospital", "label": "Organisation"},
    {"name": "Dr. Patel", "label": "Person"},
    {"name": "cardiology clinic", "label": "Organisation"},
    {"name": "Ramipril 5mg", "label": "Product"},
    {"name": "Metformin 500mg", "label": "Product"},
    {"name": "Losartan 50mg", "label": "Product"},
    {"name": "Atorvastatin 20mg", "lab
```

**All extracted entities (14):**
- Dr. Yemi Adebayo
- Mrs. Rosa Fortunato
- St. Catherine's General Hospital
- Dr. Patel
- cardiology clinic
- Ramipril 5mg
- Metformin 500mg
- Losartan 50mg
- Atorvastatin 20mg
- Atorvastatin 40mg
- Sunridge Pathology
- Hypertrophic cardiomyopathy
- Dr. Helena Marsh
- Riverside Family Practice

**All relationships (7):**
- Dr. Yemi Adebayo → prescribed → Ramipril 5mg
- Dr. Yemi Adebayo → prescribed → Metformin 500mg
- Dr. Yemi Adebayo → prescribed → Losartan 50mg
- Dr. Yemi Adebayo → prescribed → Atorvastatin 40mg
- Dr. Yemi Adebayo → prescribed → Aspirin 100mg
- Dr. Yemi Adebayo → reviewed → Mrs. Rosa Fortunato
- Dr. Yemi Adebayo → prescribed → Atorvastatin 20mg

## Legal Deposition

**Input (first 200 chars):**
```
SUPERIOR COURT OF THE STATE OF CALIFORNIA
COUNTY OF LOS ANGELES
Case No. BC-2024-087341

IN THE MATTER OF: Harlow Industries, Inc. v. Pinnacle Supply Co.

DEPOSITION OF WITNESS: Gregory Tanaka
DATE: F
```

**Raw LLM output (first 500 chars):**
```
{
  "entities": [
    {"name": "Harlow Industries, Inc.", "label": "Organisation"},
    {"name": "Pinnacle Supply Co.", "label": "Organisation"},
    {"name": "Gregory Tanaka", "label": "Person"},
    {"name": "Sandra Obi", "label": "Person"},
    {"name": "Felix Drummond", "label": "Person"},
    {"name": "Margaret Solis", "label": "Person"},
    {"name": "Collette Beaumont", "label": "Person"}
  ],
  "relationships": [
    {"subject": "Gregory Tanaka", "predicate": "employed by", "object": "Ha
```

**All extracted entities (7):**
- Harlow Industries, Inc.
- Pinnacle Supply Co.
- Gregory Tanaka
- Sandra Obi
- Felix Drummond
- Margaret Solis
- Collette Beaumont

**All relationships (9):**
- Gregory Tanaka → employed by → Harlow Industries, Inc.
- Gregory Tanaka → co-signed → purchase order
- Gregory Tanaka → reviewed → vendor
- Gregory Tanaka → signed → purchase order
- Gregory Tanaka → understood → Alameda County dispute outcome
- Harlow Industries, Inc. → had VP of Operations → Sandra Obi
- Harlow Industries, Inc. → had compliance analyst → Felix Drummond
- Gregory Tanaka → had title → Director of Procurement
- Gregory Tanaka → had knowledge of → Pinnacle Supply Co. citation

## Tech Standup

**Input (first 200 chars):**
```
Marcus: Good morning team. Let's do a quick round. Sarah, you start.

Sarah: I finished the authentication refactor on the user-service yesterday. Merged the PR into develop. Today I'm picking up the 
```

**Raw LLM output (first 500 chars):**
```
{
  "entities": [
    {"name": "Marcus", "label": "Person"},
    {"name": "Sarah", "label": "Person"},
    {"name": "Dmitri", "label": "Person"},
    {"name": "Chen Wei", "label": "Person"},
    {"name": "Jake", "label": "Person"},
    {"name": "PLAT-4521", "label": "Product"},
    {"name": "analytics database", "label": "Technology"},
    {"name": "PostgreSQL", "label": "Technology"},
    {"name": "Grafana", "label": "Technology"},
    {"name": "Terraform", "label": "Technology"},
    {"name": 
```

**All extracted entities (17):**
- Marcus
- Sarah
- Dmitri
- Chen Wei
- Jake
- PLAT-4521
- analytics database
- PostgreSQL
- Grafana
- Terraform
- AWS Lambda
- Google
- Golang
- Java
- iOS
- Android
- OOM

**All relationships (0):**

## Sales Call

**Input (first 200 chars):**
```
Rachel: Hi Tom, thanks for making time today. I'm Rachel from Nexus Analytics. I wanted to follow up on our demo last week and talk about how we can solve the reporting bottleneck your finance team is
```

**Raw LLM output (first 500 chars):**
```
{"entities":[{"name":"Rachel","label":"Person"},{"name":"Tom","label":"Person"},{"name":"David Park","label":"Person"},{"name":"Anita Krishnamurthy","label":"Person"},{"name":"Nexus Analytics","label":"Organisation"},{"name":"Snowflake","label":"Technology"},{"name":"Salesforce","label":"Technology"},{"name":"Meridian Healthcare","label":"Organisation"},{"name":"Cascade Financial Group","label":"Organisation"},{"name":"Looker","label":"Technology"},{"name":"Kevin Oduya","label":"Person"},{"name"
```

**All extracted entities (12):**
- Rachel
- Tom
- David Park
- Anita Krishnamurthy
- Nexus Analytics
- Snowflake
- Salesforce
- Meridian Healthcare
- Cascade Financial Group
- Looker
- Kevin Oduya
- Chicago

**All relationships (21):**
- Rachel → followed up → Tom
- Rachel → wants to talk about → reporting bottleneck
- Rachel → wants to talk about → integration with Snowflake warehouse
- Rachel → wants to talk about → integration with Salesforce CRM
- Rachel → wants to talk about → native Snowflake connector
- Rachel → wants to talk about → Bulk API 2.0
- Rachel → wants to talk about → unlimited dashboards
- Rachel → wants to talk about → SSO via Okta
- Rachel → wants to talk about → dedicated support
- Rachel → wants to talk about → 50% discount
- Rachel → wants to talk about → 90-day pilot
- Rachel → wants to talk about → core setup
- Rachel → wants to talk about → dedicated solutions architect
- Rachel → wants to talk about → Kevin Oduya
- Rachel → wants to talk about → Chicago office
- Rachel → wants to talk about → Meridian Healthcare
- Rachel → wants to talk about → Cascade Financial Group
- Rachel → wants to talk about → Looker
- Rachel → wants to talk about → 200 users
- Rachel → wants to talk about → 20 hours a week
- Rachel → wants to talk about → $42,0

## Podcast

**Input (first 200 chars):**
```
Host: Welcome back to The Deep Dive. I'm your host, Priya Sharma, and today I'm joined by Dr. Elena Vasquez from the Marine Biology Institute at Stanford University. Elena, you've just published a gro
```

**Raw LLM output (first 500 chars):**
```
{
  "entities": [
    {"name": "Priya Sharma", "label": "Person"},
    {"name": "Dr. Elena Vasquez", "label": "Person"},
    {"name": "Stanford University", "label": "Organisation"},
    {"name": "Marine Biology Institute", "label": "Organisation"},
    {"name": "Great Barrier Reef", "label": "Location"},
    {"name": "Acropora millepora", "label": "Technology"},
    {"name": "Nature", "label": "Publication"},
    {"name": "AIMS", "label": "Organisation"},
    {"name": "Great Barrier Reef Marine
```

**All extracted entities (17):**
- Priya Sharma
- Dr. Elena Vasquez
- Stanford University
- Marine Biology Institute
- Great Barrier Reef
- Acropora millepora
- Nature
- AIMS
- Great Barrier Reef Marine Park Authority
- National Science Foundation
- Howard Hughes Medical Institute
- Cairns
- Whitsunday Islands
- CoralWatch AI
- James Park
- Brisbane
- NOAA

**All relationships (8):**
- Dr. Elena Vasquez → published → Nature
- Dr. Elena Vasquez → studied → Acropora millepora
- Dr. Elena Vasquez → partnered with → Australian Institute of Marine Science
- Dr. Elena Vasquez → partnered with → Great Barrier Reef Marine Park Authority
- Dr. Elena Vasquez → funded → $4.2 million
- Dr. Elena Vasquez → deployed underwater monitoring stations → 12 reef sites
- Dr. Elena Vasquez → collaborated with → CoralWatch AI
- Dr. Elena Vasquez → planned expedition to → Coral Triangle

## Board Meeting

**Input (first 200 chars):**
```
MINUTES OF A MEETING OF THE BOARD OF DIRECTORS
Verdant Capital Partners Ltd.
Held at: 22 Bishopsgate, London, EC2N 4BQ
Date: 15 January 2024 | Commencement: 09:30 GMT

DIRECTORS PRESENT:
- Chairperson
```

**Raw LLM output (first 500 chars):**
```
{
  "entities": [
    {"name": "Verdant Capital Partners Ltd.", "label": "Organisation"},
    {"name": "22 Bishopsgate, London, EC2N 4BQ", "label": "Location"},
    {"name": "15 January 2024", "label": "Date"},
    {"name": "Ingrid Holmberg", "label": "Person"},
    {"name": "Kwame Asante", "label": "Person"},
    {"name": "Patricia Lim", "label": "Person"},
    {"name": "Roland Strasser", "label": "Person"},
    {"name": "Beatriz Souza", "label": "Person"},
    {"name": "Niall O'Brien", "label"
```

**All extracted entities (13):**
- Verdant Capital Partners Ltd.
- 22 Bishopsgate, London, EC2N 4BQ
- 15 January 2024
- Ingrid Holmberg
- Kwame Asante
- Patricia Lim
- Roland Strasser
- Beatriz Souza
- Niall O'Brien
- Elspeth Cairns
- Yusuf Demirci
- Fenbrook Asset Management
- Companies House

**All relationships (2):**
- Verdant Capital Partners Ltd. → held at → 22 Bishopsgate, London, EC2N 4BQ
- Verdant Capital Partners Ltd. → held on → 15 January 2024

## News Article

**Input (first 200 chars):**
```
TECH GIANT MERIDIAN SYSTEMS ANNOUNCES MAJOR RESTRUCTURING AMID ANTITRUST SCRUTINY

San Francisco — Meridian Systems announced on Thursday that it will spin off its cloud division into a separately lis
```

**Raw LLM output (first 500 chars):**
```
{
  "entities": [
    {"name": "Meridian Systems", "label": "Organisation"},
    {"name": "San Francisco", "label": "Location"},
    {"name": "Victoria Okonkwo", "label": "Person"},
    {"name": "Federal Trade Commission", "label": "Organisation"},
    {"name": "United States District Court for the Northern District of California", "label": "Location"},
    {"name": "Senator Alejandro Fuentes", "label": "Person"},
    {"name": "Washington D.C.", "label": "Location"},
    {"name": "Apex Technolog
```

**All extracted entities (15):**
- Meridian Systems
- San Francisco
- Victoria Okonkwo
- Federal Trade Commission
- United States District Court for the Northern District of California
- Senator Alejandro Fuentes
- Washington D.C.
- Apex Technology Group
- New York Stock Exchange
- Forrester Research
- Alvarez & Marsal
- Meridian Cloud Services
- DataBridge Analytics
- June
- 2024

**All relationships (8):**
- Meridian Systems → announced → spin off its cloud division into a separately listed entity
- Meridian Systems → announced → shares in Meridian Systems rose 4.2% on the New York Stock Exchange
- Meridian Systems → announced → annual general meeting in June
- Meridian Systems → announced → restructuring advisers from Alvarez & Marsal
- Meridian Systems → announced → approximately 1,200 redundancies
- Victoria Okonkwo → told reporters → press conference in San Francisco
- Victoria Okonkwo → said → Meridian Systems will spin off its cloud division into a separately listed entity
- Victoria Okonkwo → said → pending shareholder approval at the company's annual general meeting in June

