Extract the page header fields and student table rows from the image.

Return only valid JSON with this exact shape:

```json
{
  "header": {
    "school": "",
    "grade": "",
    "teacher": "",
    "Date_of_Record": "",
    "schoolyear": "",
    "semester": ""
  },
  "rows": [
    {
      "lastname": "",
      "firstname": "",
      "Sex": "",
      "age": "",
      "address": "",
      "town": "",
      "comment": "",
      "days_absent": "",
      "vaccination": "",
      "note": "",
      "Date_Left": "",
      "DoB": ""
    }
  ]
}
```

Use empty strings for unreadable or missing values. Do not include explanatory
text outside the JSON object.
