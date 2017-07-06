from collect_data import DataPuller
from orm import Base, Forum, get_mysql_session

engine, session = get_mysql_session()

def migrate_json(engine):
    Base.metadata.create_all(bind=engine)

    for d in data.forum_details.values():
        pk = int(d['pk'])
        if session.query(Forum).get(pk):
            print 'forum %s already exists in database' % d['name']
            continue

        print 'forum', d['name']

        forum = Forum(pk=pk,
                      id=d['id'],
                      name=unicode(d['name']),
                      twitter_name=unicode(d['twitterName']),
                      url=unicode(d['url']),
                      founder=int(d['founder']),
                      created_at=d['createdAt'],
                      alexa_rank=int(d.get('alexaRank', -1)),
                      category=unicode(d['category']),
                      description=unicode(d['raw_description']),
                      guidelines=unicode(d['raw_guidelines']),
                      language=str(d['language']),
                      ads_enabled=bool(d['settings']['adsEnabled']),
                      ads_video_enabled=bool(d['settings']['adsVideoEnabled']),
                      adult_content=bool(d['settings']['adultContent']),
                      allow_anon_post=bool(d['settings']['allowAnonPost']),
                      allow_anon_vote=bool(d['settings']['allowAnonVotes']),
                      allow_media=bool(d['settings']['allowMedia']),
                      disable_3rd_party_trackers=bool(d['settings'][
                          'disable3rdPartyTrackers']),
                      discovery_locked=bool(d['settings']['discoveryLocked']),
                      is_vip=bool(d['settings']['isVIP']),
                      must_verify=bool(d['settings']['mustVerify']),
                      must_verify_email=bool(d['settings']['mustVerifyEmail']),
                      discovery_enabled=bool(d['settings']['organicDiscoveryEnabled']),
                      support_level=int(d['settings']['supportLevel']),
                      unapprove_links=bool(d['settings']['unapproveLinks']))
        session.add(forum)
        session.commit()
