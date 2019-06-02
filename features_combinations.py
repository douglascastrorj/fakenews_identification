def get_combinations():
        
    combinations = [
    {
        'remove_stop_words':False,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':True,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':True,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':False,
        'remove_punct':True,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':False,
        'remove_punct':False,
        'n_gram':2,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':True,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':True,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':True,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':True,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':True
    },
    # dois ao mesmo tempo
    {
        'remove_stop_words':True,
        'stem':True,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':True,
        'stem':False,
        'remove_punct':True,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':True,
        'stem':False,
        'remove_punct':False,
        'n_gram':2,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':True,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':True,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':True,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':True,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':True,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':True,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':True,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':True,
        'ent':False
    },
    {
        'remove_stop_words':True,
        'stem':False,
        'remove_punct':False,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':True
    },
    {
        'remove_stop_words':False,
        'stem':True,
        'remove_punct':True,
        'n_gram':1,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':True,
        'remove_punct':False,
        'n_gram':2,
        'tags':False,
        'pos':False,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':True,
        'remove_punct':False,
        'n_gram':1,
        'tags':True,
        'pos':True,
        'dep':False,
        'alpha':False,
        'ent':False
    },
    {
        'remove_stop_words':False,
        'stem':True,
        'remove_punct':False,
        'n_gram':1,
        'tags':True,
        'pos':True,
        'dep':False,
        'alpha':False,
        'ent':False
    }
    ]
    return combinations