import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from src.database import init_db, get_db_session
from src.vector_store import VectorStoreManager
from src.workflow import NewsProcessingWorkflow
from src.agents.query_agent import QueryAgent
from src.models import NewsArticle, QueryRequest, NewsArticleDB, ExtractedEntityDB, StockImpactDB

# Page configuration
st.set_page_config(
    page_title="Financial News Intelligence System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .entity-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .company-badge { background-color: #e3f2fd; color: #1976d2; }
    .sector-badge { background-color: #f3e5f5; color: #7b1fa2; }
    .regulator-badge { background-color: #fff3e0; color: #f57c00; }
    .person-badge { background-color: #e8f5e9; color: #388e3c; }
    .event-badge { background-color: #fce4ec; color: #c2185b; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the system components."""
    try:
        with st.spinner("Initializing system..."):
            init_db()
            vector_store = VectorStoreManager()
            workflow = NewsProcessingWorkflow(vector_store)
            query_agent = QueryAgent(vector_store)
        return vector_store, workflow, query_agent, None
    except Exception as e:
        return None, None, None, str(e)


def get_entity_badge_html(entity_type):
    """Get HTML for entity badge."""
    badges = {
        "COMPANY": "company-badge",
        "SECTOR": "sector-badge",
        "REGULATOR": "regulator-badge",
        "PERSON": "person-badge",
        "EVENT": "event-badge"
    }
    return badges.get(entity_type, "company-badge")


def display_header():
    """Display application header."""
    st.markdown('<div class="main-header">📰 Financial News Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Multi-Agent System for Financial News Analysis</div>', unsafe_allow_html=True)


def display_statistics():
    """Display system statistics."""
    try:
        with get_db_session() as session:
            total_articles = session.query(NewsArticleDB).count()
            unique_articles = session.query(NewsArticleDB).filter(
                NewsArticleDB.is_duplicate == False
            ).count()
            duplicate_articles = session.query(NewsArticleDB).filter(
                NewsArticleDB.is_duplicate == True
            ).count()
            total_entities = session.query(ExtractedEntityDB).count()
            total_impacts = session.query(StockImpactDB).count()
            
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Articles", total_articles)
        with col2:
            st.metric("Unique Articles", unique_articles)
        with col3:
            st.metric("Duplicates", duplicate_articles)
        with col4:
            st.metric("Entities Extracted", total_entities)
        with col5:
            st.metric("Stock Impacts", total_impacts)
            
        if total_articles > 0:
            dedup_rate = (duplicate_articles / total_articles) * 100
            st.info(f"📊 Deduplication Rate: {dedup_rate:.1f}%")
        else:
            st.warning("No articles ingested yet. Please add articles using the 'Add Article' tab.")
            
    except Exception as e:
        st.warning("No data available yet. Please ingest some articles first.")


def display_entity_distribution():
    """Display entity type distribution chart."""
    try:
        with get_db_session() as session:
            entities = session.query(
                ExtractedEntityDB.entity_type,
                ExtractedEntityDB.entity_text
            ).all()
            
        if entities:
            df = pd.DataFrame(entities, columns=['Entity Type', 'Entity'])
            entity_counts = df['Entity Type'].value_counts().reset_index()
            entity_counts.columns = ['Entity Type', 'Count']
            
            fig = px.pie(
                entity_counts, 
                values='Count', 
                names='Entity Type',
                title='Entity Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entities extracted yet. Add articles to see entity distribution.")
            
    except Exception as e:
        st.error(f"Error displaying entity distribution: {e}")


def display_stock_impact_chart():
    """Display stock impact chart."""
    try:
        with get_db_session() as session:
            impacts = session.query(
                StockImpactDB.stock_symbol,
                StockImpactDB.confidence,
                StockImpactDB.impact_type
            ).all()
            
        if impacts:
            df = pd.DataFrame(impacts, columns=['Stock', 'Confidence', 'Type'])
            
            # Get top 10 most mentioned stocks
            top_stocks = df['Stock'].value_counts().head(10).reset_index()
            top_stocks.columns = ['Stock', 'Mentions']
            
            fig = px.bar(
                top_stocks,
                x='Stock',
                y='Mentions',
                title='Top 10 Most Mentioned Stocks',
                color='Mentions',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stock impacts recorded yet. Add articles to see stock analysis.")
            
    except Exception as e:
        st.error(f"Error displaying stock impacts: {e}")


def add_single_article_tab(workflow):
    """Single article addition tab."""
    st.header("📝 Add Single Article")
    
    st.info("💡 Fill in the details below to add a financial news article for processing.")
    
    with st.form("single_article_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            article_id = st.text_input(
                "Article ID*", 
                value=f"N{datetime.now().strftime('%Y%m%d%H%M%S')}",
                help="Unique identifier for the article (e.g., N1, N2, NEWS001)"
            )
        
        with col2:
            source = st.text_input(
                "Source*", 
                placeholder="e.g., Economic Times, Reuters",
                help="News source or publication name"
            )
        
        title = st.text_input(
            "Title*", 
            placeholder="Enter the article headline...",
            help="Article headline or title"
        )
        
        content = st.text_area(
            "Content*", 
            placeholder="Paste or type the full article content here...",
            height=250,
            help="Full text of the news article"
        )
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            published_date = st.date_input(
                "Published Date*", 
                value=datetime.now(),
                help="Article publication date"
            )
        
        with col2:
            published_time = st.time_input(
                "Published Time*", 
                value=datetime.now().time(),
                help="Article publication time"
            )
        
        url = st.text_input(
            "URL (Optional)", 
            placeholder="https://example.com/article",
            help="Link to the original article"
        )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button(
                "🚀 Process Article", 
                type="primary", 
                use_container_width=True
            )
        
        if submitted:
            if not article_id or not title or not content or not source:
                st.error("⚠️ Please fill in all required fields (marked with *).")
            else:
                try:
                    # Combine date and time
                    published_datetime = datetime.combine(published_date, published_time)
                    
                    # Create article
                    article = NewsArticle(
                        article_id=article_id,
                        title=title,
                        content=content,
                        source=source,
                        published_at=published_datetime,
                        url=url if url else None
                    )
                    
                    with st.spinner("🔄 Processing article through AI pipeline..."):
                        result = workflow.process_article(article)
                    
                    if result.error:
                        st.error(f"❌ Processing failed: {result.error}")
                    else:
                        st.success("✅ Article processed successfully!")
                        
                        # Display results in cards
                        st.markdown("### 📊 Processing Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if result.is_duplicate:
                                st.warning(f"⚠️ **Duplicate Detected**\n\nDuplicate of: `{result.duplicate_of}`")
                            else:
                                st.success("✓ **Unique Article**\n\nNo duplicates found")
                        with col2:
                            st.info(f"📑 **Entities Extracted**\n\n{len(result.entities)} entities found")
                        with col3:
                            st.info(f"📈 **Stock Impacts**\n\n{len(result.stock_impacts)} stocks affected")
                        
                        # Show entities
                        if result.entities and not result.is_duplicate:
                            st.markdown("### 🏷️ Extracted Entities")
                            entity_html = ""
                            for entity in result.entities:
                                badge_class = get_entity_badge_html(entity.type)
                                entity_html += f'<span class="entity-badge {badge_class}">{entity.text} ({entity.type})</span>'
                            st.markdown(entity_html, unsafe_allow_html=True)
                        
                        # Show stock impacts
                        if result.stock_impacts and not result.is_duplicate:
                            st.markdown("### 📊 Stock Impact Analysis")
                            impact_df = pd.DataFrame([
                                {
                                    "Stock Symbol": i.symbol,
                                    "Confidence": f"{i.confidence:.0%}",
                                    "Impact Type": i.type,
                                    "Reasoning": i.reasoning or "Direct mention"
                                }
                                for i in result.stock_impacts
                            ])
                            st.dataframe(impact_df, use_container_width=True, hide_index=True)
                        
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("View Error Details"):
                        st.code(str(e))


def add_bulk_articles_tab(workflow):
    """Bulk article addition tab."""
    st.header("📦 Add Multiple Articles")
    
    st.info("💡 Add multiple articles at once. Each article should be separated clearly.")
    
    # Initialize session state for articles
    if 'bulk_articles' not in st.session_state:
        st.session_state.bulk_articles = []
    
    st.markdown("### ➕ Add Articles to Batch")
    
    with st.form("bulk_article_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            article_id = st.text_input(
                "Article ID*", 
                value=f"N{len(st.session_state.bulk_articles) + 1}",
                help="Unique identifier"
            )
        
        with col2:
            source = st.text_input("Source*", placeholder="e.g., Reuters")
        
        title = st.text_input("Title*", placeholder="Article headline...")
        content = st.text_area("Content*", placeholder="Article content...", height=150)
        
        col1, col2 = st.columns(2)
        with col1:
            published_date = st.date_input("Published Date*", value=datetime.now())
        with col2:
            published_time = st.time_input("Published Time*", value=datetime.now().time())
        
        url = st.text_input("URL (Optional)", placeholder="https://...")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            add_to_batch = st.form_submit_button("➕ Add to Batch", use_container_width=True)
        
        if add_to_batch:
            if not article_id or not title or not content or not source:
                st.error("⚠️ Please fill in all required fields.")
            else:
                published_datetime = datetime.combine(published_date, published_time)
                
                article_data = {
                    'article_id': article_id,
                    'title': title,
                    'content': content,
                    'source': source,
                    'published_at': published_datetime,
                    'url': url if url else None
                }
                
                st.session_state.bulk_articles.append(article_data)
                st.success(f"✅ Added '{title}' to batch!")
                st.rerun()
    
    # Display current batch
    if st.session_state.bulk_articles:
        st.markdown(f"### 📋 Current Batch ({len(st.session_state.bulk_articles)} articles)")
        
        for idx, article_data in enumerate(st.session_state.bulk_articles):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{idx + 1}. {article_data['title']}**")
                st.caption(f"ID: {article_data['article_id']} | Source: {article_data['source']}")
            with col2:
                if st.button("🗑️ Remove", key=f"remove_{idx}"):
                    st.session_state.bulk_articles.pop(idx)
                    st.rerun()
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.bulk_articles = []
                st.rerun()
        
        with col3:
            if st.button("🚀 Process All Articles", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                for i, article_data in enumerate(st.session_state.bulk_articles):
                    status_text.text(f"Processing article {i+1}/{len(st.session_state.bulk_articles)}: {article_data['article_id']}")
                    
                    article = NewsArticle(**article_data)
                    result = workflow.process_article(article)
                    results.append((article_data, result))
                    
                    progress_bar.progress((i + 1) / len(st.session_state.bulk_articles))
                
                status_text.empty()
                progress_bar.empty()
                
                # Show results
                unique = sum(1 for _, r in results if not r.is_duplicate)
                duplicates = sum(1 for _, r in results if r.is_duplicate)
                errors = sum(1 for _, r in results if r.error)
                
                st.success(f"✅ Processed {len(results)} articles!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    st.metric("Unique Articles", unique)
                with col3:
                    st.metric("Duplicates", duplicates)
                
                if errors > 0:
                    st.error(f"⚠️ {errors} articles had processing errors")
                
                # Clear batch after processing
                st.session_state.bulk_articles = []
                st.balloons()
                st.rerun()
    else:
        st.info("👆 Add articles to the batch using the form above, then process them all at once.")


def query_tab(query_agent):
    """Query news tab."""
    st.header("🔍 Search Financial News")
    
    st.info("💡 Search for articles using natural language queries. The system understands companies, sectors, regulators, and more.")
    
    # Query examples
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - `HDFC Bank news` - Find all articles mentioning HDFC Bank
        - `Banking sector update` - Get sector-wide banking news
        - `RBI policy changes` - Find regulatory announcements
        - `Technology sector earnings` - Search for tech company results
        - `Interest rate impact` - Semantic search for related topics
        """)
    
    # Query input
    query = st.text_input(
        "🔍 Enter your search query",
        placeholder="e.g., HDFC Bank news, Banking sector update, RBI policy...",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        max_results = st.slider("Maximum Results", 1, 20, 10)
    with col2:
        include_sector = st.checkbox("Include Sector News", value=True)
    with col3:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        try:
            request = QueryRequest(
                query=query,
                max_results=max_results,
                include_sector_news=include_sector
            )
            
            with st.spinner("🔄 Searching..."):
                response = query_agent.process_query(request)
            
            st.success(f"✅ Found {response.total_results} results in {response.processing_time:.3f}s")
            
            if response.results:
                for i, article in enumerate(response.results, 1):
                    with st.expander(f"📰 {i}. {article.title}", expanded=(i == 1)):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**📰 Source:** {article.source}")
                            st.markdown(f"**📅 Published:** {article.published_at.strftime('%Y-%m-%d %H:%M')}")
                            if article.url:
                                st.markdown(f"**🔗 URL:** [{article.url}]({article.url})")
                        
                        with col2:
                            st.markdown(f"**🆔 Article ID:** `{article.article_id}`")
                            if article.is_duplicate:
                                st.warning(f"⚠️ Duplicate of: `{article.duplicate_of}`")
                        
                        # Content
                        st.markdown("**📄 Content:**")
                        content_preview = article.content[:500] + "..." if len(article.content) > 500 else article.content
                        st.text(content_preview)
                        
                        # Entities
                        if article.entities:
                            st.markdown("**🏷️ Entities:**")
                            entity_html = ""
                            for entity in article.entities[:10]:
                                badge_class = get_entity_badge_html(entity.type)
                                entity_html += f'<span class="entity-badge {badge_class}">{entity.text}</span>'
                            st.markdown(entity_html, unsafe_allow_html=True)
                        
                        # Stock impacts
                        if article.stock_impacts:
                            st.markdown("**📈 Stock Impacts:**")
                            impact_data = []
                            for impact in article.stock_impacts[:5]:
                                impact_data.append(f"**{impact.symbol}** ({impact.confidence:.0%} - {impact.type})")
                            st.markdown(" • " + " • ".join(impact_data))
            else:
                st.warning("📭 No results found. Try a different query or add more articles first.")
                
        except Exception as e:
            st.error(f"❌ Search failed: {e}")


def analytics_tab():
    """Analytics and visualization tab."""
    st.header("📊 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        display_entity_distribution()
    
    with col2:
        display_stock_impact_chart()
    
    # Timeline view
    st.markdown("### 📈 Articles Timeline")
    try:
        with get_db_session() as session:
            articles = session.query(
                NewsArticleDB.published_at,
                NewsArticleDB.is_duplicate
            ).order_by(NewsArticleDB.published_at).all()
            
        if articles:
            df = pd.DataFrame(articles, columns=['Date', 'Is Duplicate'])
            df['Date'] = pd.to_datetime(df['Date'])
            df['Count'] = 1
            
            daily_counts = df.groupby(df['Date'].dt.date)['Count'].sum().reset_index()
            daily_counts.columns = ['Date', 'Articles']
            
            fig = px.line(
                daily_counts,
                x='Date',
                y='Articles',
                title='Articles Published Over Time',
                markers=True
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for timeline visualization yet. Add more articles to see trends.")
            
    except Exception as e:
        st.info("Not enough data for timeline visualization yet.")


def main():
    """Main Streamlit application."""
    display_header()
    
    # Initialize system
    vector_store, workflow, query_agent, error = initialize_system()
    
    if error:
        st.error(f"❌ Failed to initialize system: {error}")
        st.info("Please ensure you have set up the .env file with your OPENAI_API_KEY")
        st.code("cp .env.example .env\n# Then edit .env and add your OpenAI API key")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/news.png", width=100)
        st.title("Navigation")
        
        tab_selection = st.radio(
            "Select a section:",
            ["📊 Dashboard", "📝 Add Article", "📦 Add Multiple", "🔍 Search News", "📈 Analytics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Guide")
        st.markdown("""
        1. **Add Article** - Submit single news articles
        2. **Add Multiple** - Batch process articles
        3. **Search News** - Query with natural language
        4. **Dashboard** - View statistics
        5. **Analytics** - Explore visualizations
        """)
        
        st.markdown("---")
        st.markdown("### 🤖 System Info")
        st.info("**AI-Powered Features:**\n\n✓ Duplicate Detection\n✓ Entity Extraction\n✓ Stock Impact Analysis\n✓ Semantic Search")
        
        st.markdown("---")
        st.markdown("### 📚 Tech Stack")
        st.markdown("""
        - **LangGraph** - Multi-agent orchestration
        - **OpenAI** - LLM & Embeddings
        - **ChromaDB** - Vector database
        - **spaCy** - Named Entity Recognition
        - **Streamlit** - Web interface
        """)
    
    # Main content based on tab selection
    if tab_selection == "📊 Dashboard":
        st.header("📊 System Dashboard")
        display_statistics()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            display_entity_distribution()
        with col2:
            display_stock_impact_chart()
    
    elif tab_selection == "📝 Add Article":
        add_single_article_tab(workflow)
    
    elif tab_selection == "📦 Add Multiple":
        add_bulk_articles_tab(workflow)
    
    elif tab_selection == "🔍 Search News":
        query_tab(query_agent)
    
    elif tab_selection == "📈 Analytics":
        analytics_tab()


if __name__ == "__main__":
    main()