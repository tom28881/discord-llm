"""
Personalization Engine for Discord Message Intelligence
Learns user preferences and adapts importance scoring over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger('personalization')

@dataclass
class UserInteraction:
    """Record of user interaction with a message"""
    message_id: int
    interaction_type: str  # 'view', 'react', 'reply', 'ignore', 'flag_important'
    timestamp: float
    dwell_time: float  # How long user spent looking at message (seconds)
    explicit_rating: Optional[int] = None  # User-provided importance rating (1-5)

@dataclass
class UserProfile:
    """User's personalization profile"""
    user_id: str
    interests: Dict[str, float]  # Topic -> interest weight
    activity_preferences: Dict[str, float]  # Activity type -> preference weight  
    author_preferences: Dict[str, float]  # Author ID -> preference weight
    time_preferences: Dict[int, float]  # Hour of day -> preference weight
    importance_threshold: float = 0.5  # User's threshold for "important" messages
    created_at: float = 0.0
    updated_at: float = 0.0

class UserPersonalizationEngine:
    """Engine for learning and applying user preferences"""
    
    def __init__(self, data_dir: str = "data/personalization"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Interaction tracking
        self.interactions: Dict[str, List[UserInteraction]] = defaultdict(list)
        
        # Models for preference learning
        self.preference_models: Dict[str, Any] = {}
        self.tfidf_vectorizers: Dict[str, TfidfVectorizer] = {}
        
        # Cold start handling
        self.global_patterns = {
            'popular_topics': Counter(),
            'popular_authors': Counter(),
            'popular_times': Counter(),
            'activity_engagement': Counter()
        }
        
        self.load_user_data()
    
    def record_interaction(self, user_id: str, message_id: int, 
                          interaction_type: str, dwell_time: float = 0.0,
                          explicit_rating: Optional[int] = None):
        """Record a user interaction with a message"""
        
        interaction = UserInteraction(
            message_id=message_id,
            interaction_type=interaction_type,
            timestamp=datetime.now().timestamp(),
            dwell_time=dwell_time,
            explicit_rating=explicit_rating
        )
        
        self.interactions[user_id].append(interaction)
        
        # Update user profile
        if user_id not in self.user_profiles:
            self._create_user_profile(user_id)
        
        self._update_user_profile_from_interaction(user_id, interaction)
        
        # Update global patterns for cold start
        self._update_global_patterns(interaction)
        
        logger.info(f"Recorded {interaction_type} interaction for user {user_id} on message {message_id}")
    
    def get_personalized_importance_score(self, user_id: str, message_data: Dict[str, Any],
                                        base_importance: float, base_confidence: float) -> Tuple[float, float]:
        """Get personalized importance score for a message"""
        
        if user_id not in self.user_profiles:
            # Cold start: use global patterns
            return self._cold_start_scoring(message_data, base_importance, base_confidence)
        
        profile = self.user_profiles[user_id]
        
        # Calculate personalization factors
        content_score = self._calculate_content_preference(user_id, message_data.get('content', ''))
        author_score = self._calculate_author_preference(profile, message_data.get('author_id', ''))
        time_score = self._calculate_time_preference(profile, message_data.get('sent_at', 0))
        
        # Combine factors
        personalization_multiplier = (content_score + author_score + time_score) / 3.0
        
        # Apply user's importance threshold
        threshold_adjustment = 1.0
        if base_importance < profile.importance_threshold:
            threshold_adjustment = 0.8  # Reduce importance if below user's threshold
        elif base_importance > profile.importance_threshold:
            threshold_adjustment = 1.2  # Boost importance if above threshold
        
        # Calculate final personalized score
        personalized_importance = base_importance * personalization_multiplier * threshold_adjustment
        personalized_confidence = min(base_confidence * (1.0 + personalization_multiplier * 0.2), 1.0)
        
        return float(np.clip(personalized_importance, 0.0, 1.0)), float(personalized_confidence)
    
    def update_user_preferences(self, user_id: str, feedback_data: Dict[str, Any]):
        """Update user preferences based on explicit feedback"""
        
        if user_id not in self.user_profiles:
            self._create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        # Update importance threshold if provided
        if 'importance_threshold' in feedback_data:
            profile.importance_threshold = feedback_data['importance_threshold']
        
        # Update topic preferences
        if 'topic_preferences' in feedback_data:
            for topic, weight in feedback_data['topic_preferences'].items():
                profile.interests[topic] = weight
        
        # Update activity preferences
        if 'activity_preferences' in feedback_data:
            for activity, weight in feedback_data['activity_preferences'].items():
                profile.activity_preferences[activity] = weight
        
        profile.updated_at = datetime.now().timestamp()
        self.save_user_data()
        
        logger.info(f"Updated preferences for user {user_id}")
    
    def train_preference_model(self, user_id: str, messages_df: pd.DataFrame):
        """Train personalized preference model for user"""
        
        if user_id not in self.interactions or len(self.interactions[user_id]) < 10:
            logger.info(f"Insufficient interaction data for user {user_id}")
            return
        
        # Prepare training data from interactions
        training_data = []
        labels = []
        
        for interaction in self.interactions[user_id]:
            # Find the message in dataframe
            message_row = messages_df[messages_df['id'] == interaction.message_id]
            if message_row.empty:
                continue
            
            message_data = message_row.iloc[0].to_dict()
            
            # Create feature vector
            features = self._extract_preference_features(message_data)
            training_data.append(features)
            
            # Create label based on interaction type
            label = self._interaction_to_label(interaction)
            labels.append(label)
        
        if len(training_data) < 5:
            logger.info(f"Insufficient valid training data for user {user_id}")
            return
        
        # Train model
        X = np.array(training_data)
        y = np.array(labels)
        
        # Use Passive Aggressive Regressor for online learning
        model = PassiveAggressiveRegressor(random_state=42)
        model.fit(X, y)
        
        self.preference_models[user_id] = model
        
        # Train TF-IDF model for content preferences
        content_texts = []
        content_labels = []
        
        for interaction in self.interactions[user_id]:
            message_row = messages_df[messages_df['id'] == interaction.message_id]
            if not message_row.empty:
                content = str(message_row.iloc[0].get('content', ''))
                if len(content) > 10:  # Only use substantial content
                    content_texts.append(content)
                    content_labels.append(self._interaction_to_label(interaction))
        
        if len(content_texts) > 5:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            vectorizer.fit(content_texts)
            self.tfidf_vectorizers[user_id] = vectorizer
        
        logger.info(f"Trained preference model for user {user_id} with {len(training_data)} samples")
    
    def get_recommended_activities(self, user_id: str, activities: List[Any]) -> List[Tuple[Any, float]]:
        """Rank activities by user preference"""
        
        if user_id not in self.user_profiles:
            # Cold start: rank by general engagement
            return [(activity, 0.5) for activity in activities]
        
        profile = self.user_profiles[user_id]
        ranked_activities = []
        
        for activity in activities:
            score = self._calculate_activity_preference_score(profile, activity)
            ranked_activities.append((activity, score))
        
        # Sort by preference score
        ranked_activities.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_activities
    
    def _create_user_profile(self, user_id: str):
        """Create a new user profile"""
        
        profile = UserProfile(
            user_id=user_id,
            interests={},
            activity_preferences={
                'purchase': 0.5,
                'event': 0.5,
                'consensus': 0.5,
                'discussion': 0.5
            },
            author_preferences={},
            time_preferences={hour: 0.5 for hour in range(24)},
            importance_threshold=0.5,
            created_at=datetime.now().timestamp(),
            updated_at=datetime.now().timestamp()
        )
        
        self.user_profiles[user_id] = profile
        logger.info(f"Created new user profile for {user_id}")
    
    def _update_user_profile_from_interaction(self, user_id: str, interaction: UserInteraction):
        """Update user profile based on interaction"""
        
        profile = self.user_profiles[user_id]
        
        # Update based on interaction type
        if interaction.interaction_type in ['react', 'reply', 'flag_important']:
            # Positive interaction
            engagement_boost = 0.1
        elif interaction.interaction_type == 'ignore':
            # Negative interaction
            engagement_boost = -0.05
        else:
            # Neutral interaction (view)
            engagement_boost = 0.02
        
        # Adjust based on dwell time
        if interaction.dwell_time > 10:  # Spent more than 10 seconds
            engagement_boost *= 1.5
        elif interaction.dwell_time < 2:  # Very quick view
            engagement_boost *= 0.5
        
        # Apply explicit rating if provided
        if interaction.explicit_rating is not None:
            # Rating from 1-5, convert to adjustment
            rating_adjustment = (interaction.explicit_rating - 3) * 0.1  # -0.2 to +0.2
            engagement_boost += rating_adjustment
        
        # Update time preferences
        hour = datetime.fromtimestamp(interaction.timestamp).hour
        current_pref = profile.time_preferences.get(hour, 0.5)
        profile.time_preferences[hour] = np.clip(current_pref + engagement_boost, 0.0, 1.0)
        
        profile.updated_at = datetime.now().timestamp()
    
    def _calculate_content_preference(self, user_id: str, content: str) -> float:
        """Calculate user's preference for message content"""
        
        if user_id not in self.tfidf_vectorizers or not content:
            return 0.5  # Neutral score
        
        try:
            vectorizer = self.tfidf_vectorizers[user_id]
            content_vector = vectorizer.transform([content])
            
            # Compare with user's historical preferences
            # This is a simplified approach - in production, you'd compare with
            # vectors of positively-interacted messages
            
            # For now, return based on content length and complexity
            words = content.split()
            if len(words) > 5 and len(words) < 50:  # Sweet spot
                return 0.7
            elif len(words) > 50:  # Too long
                return 0.3
            else:  # Too short
                return 0.4
            
        except Exception as e:
            logger.warning(f"Error calculating content preference: {e}")
            return 0.5
    
    def _calculate_author_preference(self, profile: UserProfile, author_id: str) -> float:
        """Calculate user's preference for message author"""
        
        if not author_id:
            return 0.5
        
        return profile.author_preferences.get(author_id, 0.5)
    
    def _calculate_time_preference(self, profile: UserProfile, timestamp: float) -> float:
        """Calculate user's preference for message timing"""
        
        if timestamp == 0:
            return 0.5
        
        hour = datetime.fromtimestamp(timestamp).hour
        return profile.time_preferences.get(hour, 0.5)
    
    def _calculate_activity_preference_score(self, profile: UserProfile, activity: Any) -> float:
        """Calculate user's preference for an activity"""
        
        activity_type = getattr(activity, 'activity_type', 'discussion')
        base_preference = profile.activity_preferences.get(activity_type, 0.5)
        
        # Adjust based on participants
        participants = getattr(activity, 'participants', set())
        author_boost = 0.0
        
        for author_id in participants:
            author_pref = profile.author_preferences.get(author_id, 0.5)
            author_boost += (author_pref - 0.5) * 0.1  # Small boost/penalty
        
        # Normalize author boost
        if len(participants) > 0:
            author_boost = author_boost / len(participants)
        
        return np.clip(base_preference + author_boost, 0.0, 1.0)
    
    def _cold_start_scoring(self, message_data: Dict[str, Any], 
                          base_importance: float, base_confidence: float) -> Tuple[float, float]:
        """Handle scoring for new users using global patterns"""
        
        # Use global popularity patterns
        author_id = message_data.get('author_id', '')
        hour = datetime.fromtimestamp(message_data.get('sent_at', 0)).hour
        
        # Boost if from popular author
        author_boost = 0.0
        if author_id in self.global_patterns['popular_authors']:
            author_popularity = self.global_patterns['popular_authors'][author_id]
            max_popularity = max(self.global_patterns['popular_authors'].values()) if self.global_patterns['popular_authors'] else 1
            author_boost = (author_popularity / max_popularity) * 0.2
        
        # Boost if at popular time
        time_boost = 0.0
        if hour in self.global_patterns['popular_times']:
            time_popularity = self.global_patterns['popular_times'][hour]
            max_time_popularity = max(self.global_patterns['popular_times'].values()) if self.global_patterns['popular_times'] else 1
            time_boost = (time_popularity / max_time_popularity) * 0.1
        
        adjusted_importance = np.clip(base_importance + author_boost + time_boost, 0.0, 1.0)
        
        return adjusted_importance, base_confidence
    
    def _extract_preference_features(self, message_data: Dict[str, Any]) -> List[float]:
        """Extract features for preference model training"""
        
        content = str(message_data.get('content', ''))
        sent_at = message_data.get('sent_at', 0)
        
        features = [
            len(content),  # Text length
            len(content.split()),  # Word count
            content.count('!'),  # Exclamation count
            content.count('?'),  # Question count
            datetime.fromtimestamp(sent_at).hour,  # Hour of day
            datetime.fromtimestamp(sent_at).weekday(),  # Day of week
        ]
        
        return features
    
    def _interaction_to_label(self, interaction: UserInteraction) -> float:
        """Convert interaction to training label"""
        
        if interaction.explicit_rating is not None:
            return interaction.explicit_rating / 5.0  # Normalize to 0-1
        
        # Convert interaction type to score
        interaction_scores = {
            'flag_important': 1.0,
            'react': 0.8,
            'reply': 0.9,
            'view': 0.4,
            'ignore': 0.1
        }
        
        base_score = interaction_scores.get(interaction.interaction_type, 0.5)
        
        # Adjust based on dwell time
        if interaction.dwell_time > 10:
            base_score *= 1.2
        elif interaction.dwell_time < 2:
            base_score *= 0.8
        
        return np.clip(base_score, 0.0, 1.0)
    
    def _update_global_patterns(self, interaction: UserInteraction):
        """Update global patterns for cold start handling"""
        
        # This would be expanded to track patterns from message data
        # For now, just track interaction types
        self.global_patterns['activity_engagement'][interaction.interaction_type] += 1
    
    def save_user_data(self):
        """Save user profiles and interactions to disk"""
        
        # Save profiles
        profiles_file = self.data_dir / "user_profiles.json"
        profiles_data = {
            user_id: asdict(profile) 
            for user_id, profile in self.user_profiles.items()
        }
        
        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2)
        
        # Save interactions
        interactions_file = self.data_dir / "user_interactions.pkl"
        with open(interactions_file, 'wb') as f:
            pickle.dump(dict(self.interactions), f)
        
        # Save models
        models_file = self.data_dir / "preference_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(self.preference_models, f)
        
        # Save TF-IDF vectorizers
        tfidf_file = self.data_dir / "tfidf_vectorizers.pkl"
        with open(tfidf_file, 'wb') as f:
            pickle.dump(self.tfidf_vectorizers, f)
        
        # Save global patterns
        patterns_file = self.data_dir / "global_patterns.pkl"
        with open(patterns_file, 'wb') as f:
            pickle.dump(self.global_patterns, f)
        
        logger.info("Saved user personalization data")
    
    def load_user_data(self):
        """Load user profiles and interactions from disk"""
        
        try:
            # Load profiles
            profiles_file = self.data_dir / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for user_id, profile_dict in profiles_data.items():
                    self.user_profiles[user_id] = UserProfile(**profile_dict)
            
            # Load interactions
            interactions_file = self.data_dir / "user_interactions.pkl"
            if interactions_file.exists():
                with open(interactions_file, 'rb') as f:
                    interactions_data = pickle.load(f)
                
                for user_id, user_interactions in interactions_data.items():
                    self.interactions[user_id] = [
                        UserInteraction(**interaction) if isinstance(interaction, dict) else interaction
                        for interaction in user_interactions
                    ]
            
            # Load models
            models_file = self.data_dir / "preference_models.pkl"
            if models_file.exists():
                with open(models_file, 'rb') as f:
                    self.preference_models = pickle.load(f)
            
            # Load TF-IDF vectorizers
            tfidf_file = self.data_dir / "tfidf_vectorizers.pkl"
            if tfidf_file.exists():
                with open(tfidf_file, 'rb') as f:
                    self.tfidf_vectorizers = pickle.load(f)
            
            # Load global patterns
            patterns_file = self.data_dir / "global_patterns.pkl"
            if patterns_file.exists():
                with open(patterns_file, 'rb') as f:
                    self.global_patterns = pickle.load(f)
            
            logger.info(f"Loaded personalization data for {len(self.user_profiles)} users")
            
        except Exception as e:
            logger.warning(f"Error loading user data: {e}")
            # Initialize empty data structures if loading fails
            self.user_profiles = {}
            self.interactions = defaultdict(list)
            self.preference_models = {}
            self.tfidf_vectorizers = {}
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's interactions and preferences"""
        
        if user_id not in self.user_profiles:
            return {"status": "new_user"}
        
        profile = self.user_profiles[user_id]
        interactions = self.interactions.get(user_id, [])
        
        interaction_counts = Counter(i.interaction_type for i in interactions)
        
        return {
            "status": "active_user",
            "profile_created": datetime.fromtimestamp(profile.created_at).isoformat(),
            "last_updated": datetime.fromtimestamp(profile.updated_at).isoformat(),
            "total_interactions": len(interactions),
            "interaction_breakdown": dict(interaction_counts),
            "importance_threshold": profile.importance_threshold,
            "top_interests": dict(Counter(profile.interests).most_common(5)),
            "preferred_activities": dict(Counter(profile.activity_preferences).most_common()),
            "has_trained_model": user_id in self.preference_models
        }