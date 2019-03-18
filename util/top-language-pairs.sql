USE wikishared;

SELECT translation_source_language, translation_target_language,
       count(translation_id) as count
FROM cx_translations
WHERE translation_status='published'
GROUP BY translation_source_language, translation_target_language
ORDER BY count DESC LIMIT 10;
