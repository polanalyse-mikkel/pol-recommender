from psycopg import sql
from psycopg.rows import dict_row

import recommender.pol_connector as pc
from recommender.schemas.recommendation_list import Recommendation, RecommendationList


class RecImpressionSanitizer:
    """
    Class for sanitizing recommendation lists for previously recommended articles.
    """

    def __init__(self, minimum_sanitzation_threshold: int = 3):
        self.number_of_cleaned_items = 0

        query = sql.SQL("""
            select subs.sso_id as user_id, article_id as item_id
            from politiken_ia_prv.layer_one_recommendations recs
            join politiken_ia_prv.subscriptions_mv subs
            on recs.sso_id = subs.sso_id_hash
            where is_impression
              and article_id != 'undefined'
            group by subs.sso_id, article_id
            having count(*) >= {threshold}""").format(
            threshold=sql.Literal(minimum_sanitzation_threshold)
        )
        with pc.connect(row_factory=dict_row) as conn:
            rows = conn.execute(query).fetchall()

        self.user_impressions = {}
        for row in rows:
            user_id = row["user_id"]
            item_id = row["item_id"]
            if user_id:
                if user_id not in self.user_impressions:
                    self.user_impressions[user_id] = set([int(item_id)])
                else:
                    self.user_impressions[user_id].add(int(item_id))

    def get_number_of_cleaned_items(self) -> int:
        return self.number_of_cleaned_items

    def __call__(
        self, user_id: str, item_list: RecommendationList
    ) -> RecommendationList:
        assert self.user_impressions != {}, "Impressions must be loaded before calling"

        # if user has no previous registered recommendation impressions, return all recommendations
        if user_id not in self.user_impressions:
            return item_list

        user_history = self.user_impressions[user_id]

        def predicate(rec: Recommendation) -> bool:
            item_id = rec.item_id
            check = item_id not in user_history
            if not check:
                self.number_of_cleaned_items += 1
            return check

        clean_item_list = item_list.filter(predicate)
        return clean_item_list
