#!/usr/bin/env python3
"""
Comprehensive Test Suite for Discord Monitoring Dashboard
Tests all ML modules and validates functionality with actual Discord data
"""

import sys
import os
import sqlite3
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import all modules to test
from lib.purchase_predictor import PurchasePredictor
from lib.deadline_tracker import DeadlineTracker
from lib.sentiment_analyzer import SentimentAnalyzer
from lib.conversation_threader import ConversationThreader

class DiscordMonitoringTester:
    """Comprehensive tester for Discord monitoring functionality"""
    
    def __init__(self, db_path: str = 'data/db.sqlite'):
        self.db_path = db_path
        self.test_results = {
            'database_tests': {},
            'ml_module_tests': {},
            'czech_language_tests': {},
            'integration_tests': {},
            'performance_tests': {}
        }
        
        # Test server ID (from requirements)
        self.test_server_id = 809811760273555506
        
        # Initialize ML modules
        try:
            self.purchase_predictor = PurchasePredictor(db_path)
            self.deadline_tracker = DeadlineTracker(db_path)
            self.sentiment_analyzer = SentimentAnalyzer(db_path)
            self.conversation_threader = ConversationThreader(db_path)
            print("✓ All ML modules initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize ML modules: {e}")
            self.purchase_predictor = None
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive report"""
        print("\n🚀 Starting Comprehensive Discord Monitoring Tests")
        print("=" * 60)
        
        # 1. Database Tests
        print("\n1. Testing Database Connectivity and Structure...")
        self.test_database_structure()
        
        # 2. ML Module Individual Tests
        print("\n2. Testing ML Modules...")
        if self.purchase_predictor:
            self.test_purchase_predictor()
            self.test_deadline_tracker()
            self.test_sentiment_analyzer()
            self.test_conversation_threader()
        
        # 3. Czech Language Support Tests
        print("\n3. Testing Czech Language Support...")
        self.test_czech_language_support()
        
        # 4. Real Data Integration Tests
        print("\n4. Testing with Real Discord Data...")
        self.test_real_data_integration()
        
        # 5. Performance Tests
        print("\n5. Testing Performance...")
        self.test_performance()
        
        # Generate final report
        return self.generate_final_report()
    
    def test_database_structure(self):
        """Test database structure and required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['servers', 'channels', 'messages']
            ml_tables = ['purchase_predictions', 'deadlines', 'sentiment_scores', 'conversation_threads']
            
            # Test basic tables
            for table in required_tables:
                if table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    self.test_results['database_tests'][f'{table}_exists'] = True
                    self.test_results['database_tests'][f'{table}_count'] = count
                    print(f"  ✓ {table}: {count:,} records")
                else:
                    self.test_results['database_tests'][f'{table}_exists'] = False
                    print(f"  ✗ {table}: Missing!")
            
            # Test ML tables (may not exist yet)
            for table in ml_tables:
                if table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    self.test_results['database_tests'][f'{table}_exists'] = True
                    self.test_results['database_tests'][f'{table}_count'] = count
                    print(f"  ✓ {table}: {count:,} records")
                else:
                    # Try to create the table
                    try:
                        self._create_missing_table(cursor, table)
                        self.test_results['database_tests'][f'{table}_exists'] = True
                        self.test_results['database_tests'][f'{table}_count'] = 0
                        print(f"  ⚠ {table}: Created (was missing)")
                    except Exception as e:
                        self.test_results['database_tests'][f'{table}_exists'] = False
                        print(f"  ✗ {table}: Failed to create - {e}")
            
            conn.commit()
            conn.close()
            
            # Test server data
            if self.test_results['database_tests'].get('messages_count', 0) > 0:
                self.test_results['database_tests']['has_test_server'] = self._check_test_server_data()
            
        except Exception as e:
            print(f"  ✗ Database test failed: {e}")
            self.test_results['database_tests']['error'] = str(e)
    
    def _create_missing_table(self, cursor, table_name: str):
        """Create missing ML tables"""
        table_schemas = {
            'purchase_predictions': '''
                CREATE TABLE purchase_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    channel_id INTEGER,
                    probability REAL,
                    prediction_type TEXT,
                    metadata TEXT,
                    predicted_at INTEGER
                )
            ''',
            'deadlines': '''
                CREATE TABLE deadlines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    channel_id INTEGER,
                    deadline_date INTEGER,
                    deadline_text TEXT,
                    urgency_level INTEGER,
                    reminder_sent INTEGER DEFAULT 0,
                    created_at INTEGER
                )
            ''',
            'sentiment_scores': '''
                CREATE TABLE sentiment_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    excitement_level REAL,
                    sentiment_type TEXT,
                    confidence REAL,
                    emoji_count INTEGER,
                    exclamation_count INTEGER,
                    analyzed_at INTEGER
                )
            ''',
            'conversation_threads': '''
                CREATE TABLE conversation_threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT,
                    message_id INTEGER,
                    channel_id INTEGER,
                    position INTEGER,
                    thread_type TEXT,
                    created_at INTEGER
                )
            '''
        }
        
        if table_name in table_schemas:
            cursor.execute(table_schemas[table_name])
    
    def _check_test_server_data(self) -> bool:
        """Check if we have data for the test server"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM messages WHERE server_id = ?", (self.test_server_id,))
            count = cursor.fetchone()[0]
            
            conn.close()
            
            if count > 0:
                print(f"  ✓ Test server data: {count:,} messages found")
                return True
            else:
                print(f"  ⚠ Test server data: No messages found for server {self.test_server_id}")
                return False
                
        except Exception as e:
            print(f"  ✗ Error checking test server data: {e}")
            return False
    
    def test_purchase_predictor(self):
        """Test purchase prediction functionality"""
        print("  Testing Purchase Predictor...")
        
        if not self.purchase_predictor:
            self.test_results['ml_module_tests']['purchase_predictor'] = {'error': 'Module not initialized'}
            return
        
        # Test with sample Czech messages
        test_messages = [
            {
                'id': 1,
                'content': 'Kdo chce společný nákup na Amazonu? Mám prime a můžeme ušetřit na poštovném',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp())
            },
            {
                'id': 2,
                'content': 'Já jo! Kolik to bude stát?',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp()) + 60
            },
            {
                'id': 3,
                'content': 'Asi 50 euro na osobu, split cost je lepší',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp()) + 120
            }
        ]
        
        try:
            result = self.purchase_predictor.predict_purchase(test_messages)
            
            self.test_results['ml_module_tests']['purchase_predictor'] = {
                'prediction_successful': True,
                'probability': result.get('probability', 0),
                'prediction_type': result.get('prediction_type', 'unknown'),
                'confidence': result.get('confidence', 0),
                'features_detected': len(result.get('features', {}))
            }
            
            print(f"    ✓ Probability: {result.get('probability', 0):.2f}")
            print(f"    ✓ Type: {result.get('prediction_type', 'unknown')}")
            print(f"    ✓ Confidence: {result.get('confidence', 0):.2f}")
            
        except Exception as e:
            print(f"    ✗ Purchase predictor failed: {e}")
            self.test_results['ml_module_tests']['purchase_predictor'] = {'error': str(e)}
    
    def test_deadline_tracker(self):
        """Test deadline tracking functionality"""
        print("  Testing Deadline Tracker...")
        
        test_messages = [
            {
                'id': 4,
                'content': 'Deadline na projekt je do pátku!',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp())
            },
            {
                'id': 5,
                'content': 'Končí to 15.3. ve 14:30',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp()) + 60
            }
        ]
        
        try:
            deadlines = self.deadline_tracker.extract_deadlines(test_messages)
            
            self.test_results['ml_module_tests']['deadline_tracker'] = {
                'deadlines_found': len(deadlines),
                'extraction_successful': True,
                'has_czech_dates': any('pátku' in str(d) or '15.3' in str(d) for d in deadlines)
            }
            
            print(f"    ✓ Deadlines found: {len(deadlines)}")
            for deadline in deadlines[:2]:  # Show first 2
                print(f"    ✓ Deadline: {deadline.get('deadline_text', 'N/A')}")
                
        except Exception as e:
            print(f"    ✗ Deadline tracker failed: {e}")
            self.test_results['ml_module_tests']['deadline_tracker'] = {'error': str(e)}
    
    def test_sentiment_analyzer(self):
        """Test sentiment analysis functionality"""
        print("  Testing Sentiment Analyzer...")
        
        test_messages = [
            {
                'id': 6,
                'content': 'To je úžasné!!! 😍🎉',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp())
            },
            {
                'id': 7,
                'content': 'Super nápad, miluju to!',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp()) + 60
            },
            {
                'id': 8,
                'content': 'To je hrozné 😞',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp()) + 120
            }
        ]
        
        try:
            result = self.sentiment_analyzer.analyze_sentiment(test_messages)
            
            self.test_results['ml_module_tests']['sentiment_analyzer'] = {
                'analysis_successful': True,
                'sentiment_score': result.get('sentiment_score', 0),
                'excitement_level': result.get('excitement_level', 0),
                'sentiment_type': result.get('sentiment_type', 'unknown'),
                'confidence': result.get('confidence', 0)
            }
            
            print(f"    ✓ Sentiment score: {result.get('sentiment_score', 0):.2f}")
            print(f"    ✓ Excitement level: {result.get('excitement_level', 0):.2f}")
            print(f"    ✓ Type: {result.get('sentiment_type', 'unknown')}")
            
        except Exception as e:
            print(f"    ✗ Sentiment analyzer failed: {e}")
            self.test_results['ml_module_tests']['sentiment_analyzer'] = {'error': str(e)}
    
    def test_conversation_threader(self):
        """Test conversation threading functionality"""
        print("  Testing Conversation Threader...")
        
        test_messages = [
            {
                'id': 9,
                'content': 'Máte někdo čas na meeting zítra?',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp())
            },
            {
                'id': 10,
                'content': 'Já můžu od 14:00',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp()) + 60
            },
            {
                'id': 11,
                'content': 'Taky můžu, kde se potkáme?',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp()) + 120
            }
        ]
        
        try:
            threads = self.conversation_threader.thread_messages(test_messages)
            
            self.test_results['ml_module_tests']['conversation_threader'] = {
                'threading_successful': True,
                'threads_created': len(threads),
                'total_messages': sum(len(t.get('messages', [])) for t in threads)
            }
            
            print(f"    ✓ Threads created: {len(threads)}")
            for i, thread in enumerate(threads[:2]):  # Show first 2
                print(f"    ✓ Thread {i+1}: {len(thread.get('messages', []))} messages, type: {thread.get('thread_type', 'unknown')}")
                
        except Exception as e:
            print(f"    ✗ Conversation threader failed: {e}")
            self.test_results['ml_module_tests']['conversation_threader'] = {'error': str(e)}
    
    def test_czech_language_support(self):
        """Test Czech language pattern recognition across all modules"""
        print("  Testing Czech Language Support...")
        
        czech_test_cases = [
            {
                'text': 'Společný nákup na Aliexpressu, kdo má zájem? Končí do pátku!',
                'expected': {
                    'purchase': True,
                    'deadline': True,
                    'urgency': True
                }
            },
            {
                'text': 'Super nápad! Já jo, kolik to bude? 💪',
                'expected': {
                    'positive_sentiment': True,
                    'excitement': True,
                    'participation': True
                }
            }
        ]
        
        czech_results = {}
        
        for i, case in enumerate(czech_test_cases):
            case_results = {}
            
            # Test purchase predictor
            if self.purchase_predictor:
                test_msg = [{
                    'id': i + 100,
                    'content': case['text'],
                    'channel_id': 123,
                    'sent_at': int(datetime.now().timestamp())
                }]
                
                try:
                    purchase_result = self.purchase_predictor.predict_purchase(test_msg)
                    case_results['purchase_detected'] = purchase_result.get('probability', 0) > 0.3
                    case_results['purchase_probability'] = purchase_result.get('probability', 0)
                except Exception as e:
                    case_results['purchase_error'] = str(e)
                
                # Test deadline tracker
                try:
                    deadline_result = self.deadline_tracker.extract_deadlines(test_msg)
                    case_results['deadline_detected'] = len(deadline_result) > 0
                    case_results['deadlines_count'] = len(deadline_result)
                except Exception as e:
                    case_results['deadline_error'] = str(e)
                
                # Test sentiment analyzer
                try:
                    sentiment_result = self.sentiment_analyzer.analyze_sentiment(test_msg)
                    case_results['sentiment_positive'] = sentiment_result.get('sentiment_score', 0) > 0
                    case_results['excitement_high'] = sentiment_result.get('excitement_level', 0) > 0.3
                except Exception as e:
                    case_results['sentiment_error'] = str(e)
            
            czech_results[f'case_{i+1}'] = case_results
            
        self.test_results['czech_language_tests'] = czech_results
        
        # Summary
        working_modules = 0
        total_modules = 3
        for case in czech_results.values():
            if case.get('purchase_detected') is not None:
                working_modules += 1
                break
        
        print(f"    ✓ Czech language support: {working_modules}/{total_modules} modules working")
    
    def test_real_data_integration(self):
        """Test with real Discord data if available"""
        print("  Testing Real Data Integration...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent messages from test server if available
            cursor.execute("""
                SELECT id, content, channel_id, sent_at
                FROM messages 
                WHERE server_id = ? 
                ORDER BY sent_at DESC 
                LIMIT 100
            """, (self.test_server_id,))
            
            real_messages = []
            for row in cursor.fetchall():
                real_messages.append({
                    'id': row[0],
                    'content': row[1] or '',
                    'channel_id': row[2],
                    'sent_at': row[3]
                })
            
            conn.close()
            
            if not real_messages:
                self.test_results['integration_tests']['real_data'] = {
                    'status': 'no_data',
                    'message': 'No real data available for testing'
                }
                print("    ⚠ No real data available for integration testing")
                return
            
            # Test all modules with real data
            integration_results = {}
            
            if self.purchase_predictor:
                try:
                    # Test recent messages for purchases
                    purchase_results = []
                    batch_size = 10
                    for i in range(0, min(len(real_messages), 50), batch_size):
                        batch = real_messages[i:i+batch_size]
                        result = self.purchase_predictor.predict_purchase(batch)
                        if result.get('probability', 0) > 0.5:
                            purchase_results.append(result)
                    
                    integration_results['purchase_predictor'] = {
                        'messages_tested': min(len(real_messages), 50),
                        'predictions_made': len(purchase_results),
                        'high_probability_found': len([r for r in purchase_results if r.get('probability', 0) > 0.7])
                    }
                    
                    print(f"    ✓ Purchase analysis: {len(purchase_results)} potential purchases found")
                    
                except Exception as e:
                    integration_results['purchase_predictor'] = {'error': str(e)}
                    print(f"    ✗ Purchase analysis failed: {e}")
            
            # Test deadline extraction
            try:
                deadline_results = self.deadline_tracker.extract_deadlines(real_messages[:30])
                integration_results['deadline_tracker'] = {
                    'messages_tested': min(len(real_messages), 30),
                    'deadlines_found': len(deadline_results)
                }
                print(f"    ✓ Deadline analysis: {len(deadline_results)} deadlines found")
            except Exception as e:
                integration_results['deadline_tracker'] = {'error': str(e)}
                print(f"    ✗ Deadline analysis failed: {e}")
            
            # Test sentiment analysis
            try:
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(real_messages[:20])
                integration_results['sentiment_analyzer'] = {
                    'messages_tested': min(len(real_messages), 20),
                    'sentiment_score': sentiment_result.get('sentiment_score', 0),
                    'excitement_level': sentiment_result.get('excitement_level', 0)
                }
                print(f"    ✓ Sentiment analysis: {sentiment_result.get('sentiment_type', 'unknown')} mood detected")
            except Exception as e:
                integration_results['sentiment_analyzer'] = {'error': str(e)}
                print(f"    ✗ Sentiment analysis failed: {e}")
            
            self.test_results['integration_tests']['real_data'] = integration_results
            
        except Exception as e:
            print(f"    ✗ Real data integration test failed: {e}")
            self.test_results['integration_tests']['real_data'] = {'error': str(e)}
    
    def test_performance(self):
        """Test performance with larger datasets"""
        print("  Testing Performance...")
        
        # Generate test data
        large_test_messages = []
        for i in range(1000):
            large_test_messages.append({
                'id': i + 1000,
                'content': f'Test message {i} with some content to analyze',
                'channel_id': 123,
                'sent_at': int(datetime.now().timestamp()) + i * 60
            })
        
        performance_results = {}
        
        if self.purchase_predictor:
            # Test batch processing
            try:
                start_time = datetime.now()
                
                batch_size = 50
                results = []
                for i in range(0, len(large_test_messages), batch_size):
                    batch = large_test_messages[i:i+batch_size]
                    result = self.purchase_predictor.predict_purchase(batch)
                    results.append(result)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                performance_results['purchase_predictor'] = {
                    'messages_processed': len(large_test_messages),
                    'processing_time_seconds': processing_time,
                    'messages_per_second': len(large_test_messages) / processing_time if processing_time > 0 else 0,
                    'batches_processed': len(results)
                }
                
                print(f"    ✓ Purchase predictor: {len(large_test_messages)} messages in {processing_time:.2f}s")
                
            except Exception as e:
                performance_results['purchase_predictor'] = {'error': str(e)}
                print(f"    ✗ Purchase predictor performance test failed: {e}")
        
        self.test_results['performance_tests'] = performance_results
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        # Calculate overall health score
        health_score = self._calculate_health_score()
        
        # Database summary
        print(f"\n📁 DATABASE STATUS:")
        db_tests = self.test_results['database_tests']
        message_count = db_tests.get('messages_count', 0)
        print(f"  Messages in database: {message_count:,}")
        print(f"  Test server data: {'✓' if db_tests.get('has_test_server', False) else '✗'}")
        print(f"  ML tables ready: {'✓' if all(db_tests.get(f'{t}_exists', False) for t in ['purchase_predictions', 'deadlines', 'sentiment_scores', 'conversation_threads']) else '⚠'}")
        
        # ML Modules summary
        print(f"\n🤖 ML MODULES STATUS:")
        ml_tests = self.test_results['ml_module_tests']
        for module, results in ml_tests.items():
            if 'error' in results:
                print(f"  {module}: ✗ {results['error']}")
            else:
                print(f"  {module}: ✓ Working")
        
        # Czech language summary
        print(f"\n🇨🇿 CZECH LANGUAGE SUPPORT:")
        czech_tests = self.test_results['czech_language_tests']
        working_cases = 0
        for case in czech_tests.values():
            if isinstance(case, dict):
                has_error = any('error' in key for key in case.keys())
                if not has_error:
                    working_cases += 1
        total_cases = len(czech_tests)
        print(f"  Test cases working: {working_cases}/{total_cases}")
        
        # Integration summary
        print(f"\n🔗 REAL DATA INTEGRATION:")
        integration_tests = self.test_results.get('integration_tests', {})
        real_data = integration_tests.get('real_data', {})
        if real_data.get('status') == 'no_data':
            print(f"  Status: ⚠ {real_data.get('message', 'No data available')}")
        elif 'error' in real_data:
            print(f"  Status: ✗ {real_data['error']}")
        else:
            working_modules = len([k for k, v in real_data.items() if isinstance(v, dict) and 'error' not in v])
            total_modules = len([k for k, v in real_data.items() if isinstance(v, dict)])
            print(f"  Modules working with real data: {working_modules}/{total_modules}")
        
        # Performance summary
        print(f"\n⚡ PERFORMANCE:")
        perf_tests = self.test_results.get('performance_tests', {})
        if perf_tests:
            for module, results in perf_tests.items():
                if 'error' not in results:
                    msg_per_sec = results.get('messages_per_second', 0)
                    print(f"  {module}: {msg_per_sec:.1f} messages/second")
        
        # Overall assessment
        print(f"\n🎯 OVERALL HEALTH SCORE: {health_score}/100")
        if health_score >= 80:
            print("  Status: ✅ EXCELLENT - All systems operational")
        elif health_score >= 60:
            print("  Status: ✅ GOOD - Minor issues detected")
        elif health_score >= 40:
            print("  Status: ⚠ FAIR - Some functionality limited")
        else:
            print("  Status: ❌ POOR - Significant issues require attention")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            print(f"  • {rec}")
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'detailed_results': self.test_results,
            'recommendations': recommendations,
            'summary': {
                'database_ready': message_count > 0,
                'ml_modules_working': len([m for m in ml_tests.values() if 'error' not in m]),
                'czech_support_working': working_cases > 0,
                'real_data_available': db_tests.get('has_test_server', False),
                'performance_acceptable': any(r.get('messages_per_second', 0) > 10 for r in perf_tests.values())
            }
        }
        
        return final_report
    
    def _calculate_health_score(self) -> int:
        """Calculate overall system health score (0-100)"""
        score = 0
        
        # Database health (30 points)
        db_tests = self.test_results['database_tests']
        if db_tests.get('messages_count', 0) > 0:
            score += 15
        if db_tests.get('has_test_server', False):
            score += 10
        if all(db_tests.get(f'{t}_exists', False) for t in ['purchase_predictions', 'deadlines', 'sentiment_scores', 'conversation_threads']):
            score += 5
        
        # ML modules health (40 points)
        ml_tests = self.test_results['ml_module_tests']
        working_modules = len([m for m in ml_tests.values() if 'error' not in m])
        total_modules = len(ml_tests) if ml_tests else 4
        score += int((working_modules / max(total_modules, 1)) * 40)
        
        # Czech language support (15 points)
        czech_tests = self.test_results['czech_language_tests']
        if czech_tests:
            working_cases = 0
            for case in czech_tests.values():
                if isinstance(case, dict):
                    has_error = any('error' in key for key in case.keys())
                    if not has_error:
                        working_cases += 1
            total_cases = len(czech_tests)
            score += int((working_cases / max(total_cases, 1)) * 15)
        
        # Integration (10 points)
        integration_tests = self.test_results.get('integration_tests', {})
        real_data = integration_tests.get('real_data', {})
        if isinstance(real_data, dict) and 'error' not in real_data and real_data.get('status') != 'no_data':
            working_modules = len([k for k, v in real_data.items() if isinstance(v, dict) and 'error' not in v])
            if working_modules > 0:
                score += 10
        
        # Performance (5 points)
        perf_tests = self.test_results.get('performance_tests', {})
        if any(r.get('messages_per_second', 0) > 5 for r in perf_tests.values()):
            score += 5
        
        return min(score, 100)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        db_tests = self.test_results['database_tests']
        ml_tests = self.test_results['ml_module_tests']
        
        # Database recommendations
        if db_tests.get('messages_count', 0) == 0:
            recommendations.append("Import Discord messages using 'python load_messages.py' to populate the database")
        
        if not db_tests.get('has_test_server', False):
            recommendations.append(f"Import messages from test server {self.test_server_id} for comprehensive testing")
        
        # ML module recommendations
        for module, results in ml_tests.items():
            if 'error' in results:
                recommendations.append(f"Fix {module}: {results['error']}")
        
        # Integration recommendations
        integration_tests = self.test_results.get('integration_tests', {})
        real_data = integration_tests.get('real_data', {})
        if real_data.get('status') == 'no_data':
            recommendations.append("Load real Discord data to test ML modules with actual conversations")
        
        # Performance recommendations
        perf_tests = self.test_results.get('performance_tests', {})
        if not any(r.get('messages_per_second', 0) > 10 for r in perf_tests.values()):
            recommendations.append("Consider optimizing ML algorithms for better performance with large datasets")
        
        if not recommendations:
            recommendations.append("All systems are functioning well! Consider monitoring dashboard at http://localhost:8502")
        
        return recommendations


def main():
    """Run comprehensive tests"""
    print("🧪 Discord Monitoring Comprehensive Test Suite")
    print("Testing all ML modules and dashboard functionality...")
    
    # Check if database exists
    db_path = 'data/db.sqlite'
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        print("Please run 'python load_messages.py' first to create and populate the database")
        return
    
    tester = DiscordMonitoringTester(db_path)
    
    try:
        final_report = tester.run_comprehensive_tests()
        
        # Save report to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return final_report
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    main()